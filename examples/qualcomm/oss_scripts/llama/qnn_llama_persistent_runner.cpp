/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model is serialized using the flatbuffer format. If specified, seq_len can exceed the context length defined in the model.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");
DEFINE_string(
    system_prompt,
    "",
    "Default system prompt for interactive mode and fallback service requests.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
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
    "[Lookahead Decoding] Represents the size of the n-grams used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Determines how many future tokens the algorithm attempts to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.");
DEFINE_bool(
    interactive,
    false,
    "Run as a persistent REPL. Each input line is treated as a prompt. Type /quit to exit.");
DEFINE_bool(
    stdio_server,
    false,
    "Run as a persistent stdio service for the companion HTTP API shim.");

namespace {

using executorch::extension::Module;
using executorch::extension::llm::GenerationConfig;
using executorch::runtime::Error;
using executorch::runtime::Result;

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

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::DecoderModelVersion decoder_model_version) {
  std::string formatted_prompt;
  switch (decoder_model_version) {
    case example::DecoderModelVersion::kLlama2:
    case example::DecoderModelVersion::kQwen2_5:
    case example::DecoderModelVersion::kCodegen:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case example::DecoderModelVersion::kGemma:
    case example::DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case example::DecoderModelVersion::kGemma2:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      break;
    case example::DecoderModelVersion::kGranite:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end_of_text|>\n");
      }
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>assistant<|end_of_role|>");
      break;
    case example::DecoderModelVersion::kPhi4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end|>");
      }
      formatted_prompt.append("<|user|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end|><|assistant|>");
      break;
    case example::DecoderModelVersion::kQwen3:
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>assistant");
      break;
    case example::DecoderModelVersion::kSmollm2_135m:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n\n");
      break;
    case example::DecoderModelVersion::kSmollm3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("\n\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    case example::DecoderModelVersion::kGlm:
      formatted_prompt.append("<|user|>\n");
      formatted_prompt.append(prompt);
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>\n");
        formatted_prompt.append(system_prompt);
      }
      formatted_prompt.append("<|assistant|>\n");
      break;
    default:
      ET_CHECK_MSG(false, "unsupported llama version");
      break;
  }
  return formatted_prompt;
}

std::string run_prompt(
    example::Runner& runner,
    example::DecoderModelVersion decoder_model_version,
    const std::string& prompt,
    const std::string& system_prompt,
    int32_t seq_len) {
  runner.reset();

  const std::string formatted_prompt =
      get_formatted_prompt(prompt, system_prompt, decoder_model_version);
  std::string response;
  GenerationConfig config{
      false,
      "",
      "",
      false,
      -1,
      false,
      seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  auto callback = [&](const std::string& piece) { response.append(piece); };
  const Error error =
      runner.generate_from_prompt_or_file(formatted_prompt, false, config, callback);
  ET_CHECK_MSG(error == Error::Ok, "Generation failed with error code %d", static_cast<int>(error));
  sanitize_response(&response);
  return response;
}

bool read_sized_payload(std::istream& input, size_t size, std::string* payload) {
  payload->assign(size, '\0');
  input.read(payload->data(), static_cast<std::streamsize>(size));
  return input.good();
}

void write_service_response(const std::string& status, const std::string& payload) {
  std::cout << status << ' ' << payload.size() << '\n';
  if (!payload.empty()) {
    std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
  std::cout.flush();
}

void run_stdio_service(
    example::Runner& runner,
    example::DecoderModelVersion decoder_model_version) {
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
    if (!(header_stream >> prompt_size >> system_prompt_size >> seq_len)) {
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

    const std::string response =
        run_prompt(runner, decoder_model_version, prompt, system_prompt, seq_len);
    write_service_response("OK", response);
  }
}

void run_interactive_repl(
    example::Runner& runner,
    example::DecoderModelVersion decoder_model_version) {
  std::cerr << "Persistent runner ready. Type /quit to exit." << std::endl;
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

    const std::string response =
        run_prompt(runner, decoder_model_version, prompt, FLAGS_system_prompt, FLAGS_seq_len);
    std::cout << response << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ET_CHECK_MSG(
      FLAGS_stdio_server || FLAGS_interactive,
      "Specify either --stdio_server or --interactive");

  std::unique_ptr<Module> module = std::make_unique<Module>(
      FLAGS_model_path.c_str(),
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module = std::make_unique<Module>(
        FLAGS_attention_sink_rope_path.c_str(),
        Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  example::Runner runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      "",
      "",
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      nullptr,
      std::move(attention_sink_rope_module));

  Result<example::DecoderModelVersion> decoder_model_version =
      runner.get_decoder_model_version();
  ET_CHECK_MSG(
      decoder_model_version.error() == Error::Ok,
      "Failed to initialize persistent runner: error code %d",
      static_cast<int>(decoder_model_version.error()));

  if (FLAGS_stdio_server) {
    run_stdio_service(runner, decoder_model_version.get());
  } else {
    run_interactive_repl(runner, decoder_model_version.get());
  }

  return 0;
}