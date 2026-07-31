/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/chat_template.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/utils.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::llm::GenerationConfig;
using executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using executorch::extension::llm::MultimodalInput;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

DEFINE_string(
    tok_embedding_path,
    "tok_embedding.pte",
    "Path to tok_embedding model serialized in flatbuffer format.");
DEFINE_string(
    encoder_path,
    "encoder.pte",
    "Path to vision encoder model serialized in flatbuffer format.");
DEFINE_string(
    decoder_path,
    "decoder.pte",
    "Path to decoder model serialized in flatbuffer format.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");
DEFINE_string(
    decoder_model_version,
    "internvl3",
    "The decoder model version to execute.");
DEFINE_string(
    system_prompt,
    "",
    "Default system prompt for interactive mode and fallback service requests.");
DEFINE_string(
    image_path,
    "",
    "Optional path to an input-list file containing preprocessed raw image tensors for interactive mode.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    1024,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy use case between the application and device/co-processor associated with the backend.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] Size of n-grams used in lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Number of future tokens to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Maximum number of speculations or candidate n-grams.");
DEFINE_bool(
    interactive,
    false,
    "Run as a persistent REPL. Each input line is treated as a prompt. --image_path can pin a preprocessed image input-list for all turns.");
DEFINE_bool(
    stdio_server,
    false,
    "Run as a persistent stdio service for the companion HTTP API shim.");

namespace {

void trim_suffix(std::string* text, const std::string& suffix) {
  while (text->size() >= suffix.size() &&
         text->compare(text->size() - suffix.size(), suffix.size(), suffix) == 0) {
    text->erase(text->size() - suffix.size());
  }
}

void trim_ascii_whitespace(std::string* text) {
  while (!text->empty()) {
    const char last = text->back();
    if (last != ' ' && last != '\n' && last != '\r' && last != '\t') {
      break;
    }
    text->pop_back();
  }
}

void sanitize_response(std::string* response) {
  static constexpr const char* kSuffixes[] = {
      "<|eot_id|>",
      "<|im_end|>",
      "<|end_of_text|>",
      "<end_of_turn>",
      "<|end|>",
      "<|endoftext|>",
  };

  bool removed_suffix = true;
  while (removed_suffix) {
    removed_suffix = false;
    trim_ascii_whitespace(response);
    for (const char* suffix : kSuffixes) {
      const std::string suffix_string(suffix);
      if (response->size() >= suffix_string.size() &&
          response->compare(
              response->size() - suffix_string.size(),
              suffix_string.size(),
              suffix_string) == 0) {
        trim_suffix(response, suffix_string);
        removed_suffix = true;
      }
    }
  }
  trim_ascii_whitespace(response);
}

bool read_sized_payload(std::istream& input, size_t size, std::string* payload) {
  payload->assign(size, '\0');
  input.read(payload->data(), static_cast<std::streamsize>(size));
  return input.good();
}

bool read_size_line(std::istream& input, size_t* size) {
  std::string line;
  if (!std::getline(input, line)) {
    return false;
  }
  if (line.empty()) {
    return false;
  }
  std::istringstream size_stream(line);
  return static_cast<bool>(size_stream >> *size);
}

void write_service_response(const std::string& status, const std::string& payload) {
  std::cout << status << ' ' << payload.size() << '\n';
  if (!payload.empty()) {
    std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
  std::cout.flush();
}

std::string run_prompt(
    example::QNNMultimodalRunner& runner,
    example::ModelVersion model_version,
    const std::vector<int32_t>& expected_size,
    ScalarType expected_dtype,
    const std::string& prompt,
    const std::string& system_prompt,
    const std::vector<std::string>& image_paths,
    int32_t seq_len,
    bool continue_conversation) {
  if (!continue_conversation) {
    runner.reset();
  }

  std::vector<std::string> prompts{prompt};
  std::vector<Message> messages = prepare_messages(prompts, image_paths, {});
  ET_CHECK_MSG(messages.size() == 1, "Expected a single prompt request");

  const Message& message = messages.front();
  std::vector<MultimodalInput> inputs;
  for (const std::string& file_path : message.files_path) {
    Image image;
    example::load_image(file_path, image, expected_size, expected_dtype);
    inputs.emplace_back(make_image_input(image));
  }

  std::string formatted_prompt =
      apply_chat_template(message.text, system_prompt, model_version);
  inputs.emplace_back(make_text_input(formatted_prompt));
  inputs = dispatch_inputs(inputs, formatted_prompt);

  std::string response;
  GenerationConfig config;
  config.echo = false;
  config.ignore_eos = false;
  config.max_new_tokens = -1;
  config.warming = false;
  config.seq_len = seq_len;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.num_bos = 0;
  config.num_eos = 0;

  auto callback = [&](const std::string& piece) { response.append(piece); };
  const Error error = runner.generate(inputs, config, callback);
  ET_CHECK_MSG(
      error == Error::Ok,
      "Generation failed with error code %d",
      static_cast<int>(error));
  sanitize_response(&response);
  return response;
}

void run_stdio_service(
    example::QNNMultimodalRunner& runner,
    example::ModelVersion model_version,
    const std::vector<int32_t>& expected_size,
    ScalarType expected_dtype) {
  std::cout << "READY" << std::endl;

  std::string header;
  while (std::getline(std::cin, header)) {
    if (header.empty()) {
      continue;
    }
    if (header == "QUIT") {
      break;
    }

    std::istringstream header_stream(header);
    size_t prompt_size = 0;
    size_t system_prompt_size = 0;
    int32_t seq_len = FLAGS_seq_len;
    size_t image_count = 0;
    int32_t continue_flag = 0;
    if (!(header_stream >> prompt_size >> system_prompt_size >> seq_len >> image_count >> continue_flag)) {
      write_service_response("ERR", "invalid request header");
      continue;
    }

    std::string prompt;
    std::string system_prompt;
    if (!read_sized_payload(std::cin, prompt_size, &prompt) ||
        !read_sized_payload(std::cin, system_prompt_size, &system_prompt)) {
      write_service_response("ERR", "unexpected end of request payload");
      break;
    }

    std::vector<std::string> image_paths;
    image_paths.reserve(image_count);
    bool image_read_failed = false;
    for (size_t index = 0; index < image_count; ++index) {
      size_t image_path_size = 0;
      if (!read_size_line(std::cin, &image_path_size)) {
        image_read_failed = true;
        break;
      }
      std::string image_path;
      if (!read_sized_payload(std::cin, image_path_size, &image_path)) {
        image_read_failed = true;
        break;
      }
      image_paths.push_back(std::move(image_path));
    }
    if (image_read_failed) {
      write_service_response("ERR", "unexpected end of image path payload");
      break;
    }

    if (prompt.empty()) {
      write_service_response("ERR", "prompt must not be empty");
      continue;
    }
    if (seq_len <= 0) {
      write_service_response("ERR", "seq_len must be greater than zero");
      continue;
    }
    if (system_prompt.empty()) {
      system_prompt = FLAGS_system_prompt;
    }

    const std::string response = run_prompt(
        runner,
        model_version,
        expected_size,
        expected_dtype,
        prompt,
        system_prompt,
        image_paths,
        seq_len,
        continue_flag != 0);
    write_service_response("OK", response);
  }
}

void run_interactive_repl(
    example::QNNMultimodalRunner& runner,
    example::ModelVersion model_version,
    const std::vector<int32_t>& expected_size,
    ScalarType expected_dtype) {
  std::vector<std::string> image_paths;
  if (!FLAGS_image_path.empty()) {
    image_paths = example::load_raw_files(FLAGS_image_path.c_str());
  }

  std::cerr << "Persistent multimodal runner ready. Type /quit to exit." << std::endl;
  std::string prompt;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, prompt)) {
      break;
    }
    if (prompt == "/quit") {
      break;
    }
    if (prompt.empty()) {
      continue;
    }

    const std::string response = run_prompt(
        runner,
        model_version,
        expected_size,
        expected_dtype,
        prompt,
        FLAGS_system_prompt,
        image_paths,
        FLAGS_seq_len,
        false);
    std::cout << response << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ET_CHECK_MSG(
      FLAGS_stdio_server || FLAGS_interactive,
      "Specify either --stdio_server or --interactive");

  std::unique_ptr<Module> encoder = std::make_unique<Module>(
      FLAGS_encoder_path.c_str(),
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<Module> tok_embedding = std::make_unique<Module>(
      FLAGS_tok_embedding_path.c_str(),
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<Module> text_decoder = std::make_unique<Module>(
      FLAGS_decoder_path.c_str(),
      Module::LoadMode::MmapUseMlockIgnoreErrors);

  example::QNNMultimodalRunner runner(
      std::move(encoder),
      std::move(tok_embedding),
      std::move(text_decoder),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_tokenizer_path.c_str(),
      "",
      "",
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap);

  Result<example::ModelVersion> model_version = runner.get_model_version();
  ET_CHECK_MSG(
      model_version.error() == Error::Ok,
      "Failed to initialize persistent multimodal runner: error code %d",
      static_cast<int>(model_version.error()));

  Result<MethodMeta> method_meta = runner.get_encoder_method_meta();
  ET_CHECK_MSG(
      method_meta.error() == Error::Ok,
      "Failed to load encoder metadata: error code %d",
      static_cast<int>(method_meta.error()));
  auto input_meta_result = method_meta->input_tensor_meta(0);
  std::vector<int32_t> expected_size(
      input_meta_result->sizes().begin(), input_meta_result->sizes().end());
  ScalarType expected_dtype = input_meta_result->scalar_type();

  if (FLAGS_stdio_server) {
    run_stdio_service(
        runner,
        model_version.get(),
        expected_size,
        expected_dtype);
  } else {
    run_interactive_repl(
        runner,
        model_version.get(),
        expected_size,
        expected_dtype);
  }

  return 0;
}