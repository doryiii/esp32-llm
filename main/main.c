/* Copyright (c) 2026 Dory
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef CONFIG_LLM_INFERENCE_INT8
#include "llm8.h"
#else
#include "llm.h"
#endif

#define PIN_LCD_BL 14

#define TAG "main"


void print_token(const char* token, void* user_data) {
  printf("%s", token);
  fflush(stdout);
}

void run_llm(char* checkpoint_path, char* tokenizer_path) {
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 512;
  char* prompt = "";
  unsigned long long rng_seed = (unsigned int)time(NULL);

  printf("\n========== LLM Inference Test ==========\n");
  ESP_LOGI(
      TAG, "Free heap (Internal): %zu bytes, (PSRAM): %zu bytes",
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM)
  );

  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);

  ESP_LOGI(
      TAG,
      "Transformer built. Free heap (Internal): %zu, (PSRAM): %zu",
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM)
  );
  ESP_LOGI(
      TAG, "Heap low watermark: %zu",
      heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT)
  );
  ESP_LOGI(TAG, "model: %s, tokenizer %s", checkpoint_path, tokenizer_path);

  if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;

  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  Sampler sampler;
  build_sampler(
      &sampler, transformer.config.vocab_size, temperature, topp, rng_seed
  );

  ESP_LOGI(TAG, "Starting generation...");
  generate(
      &transformer, &tokenizer, &sampler, prompt, steps, NULL, 0, print_token,
      NULL
  );

  // Free allocated memory
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);

  ESP_LOGI(
      TAG,
      "Generation finished. Free heap (Internal): %zu, (PSRAM): %zu",
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM)
  );
  ESP_LOGI(
      TAG, "Heap low watermark: %zu",
      heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT)
  );
  printf("========================================\n\n");
}


void run_echo_loop(void) {
  printf("Press Enter to start LLM generation...\n");

  while (1) {
    int c = getchar();
    if (c == EOF || c == 0xFF) {
      vTaskDelay(1);
      continue;
    }

    if (c == '\n' || c == '\r') {
#ifdef CONFIG_LLM_INFERENCE_INT8
      run_llm(CONFIG_LLM_MODEL_PATH_INT8, CONFIG_LLM_TOKENIZER_PATH_INT8);
#else
      run_llm(CONFIG_LLM_MODEL_PATH_FLOAT32, CONFIG_LLM_TOKENIZER_PATH_FLOAT32);
#endif
      printf("\nPress Enter to run again...\n");
    }

    vTaskDelay(1);
  }
}


void init_spiffs(void) {
  esp_vfs_spiffs_conf_t conf = {
      .base_path = "/spiffs",
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = true
  };

  esp_err_t ret = esp_vfs_spiffs_register(&conf);
  if (ret != ESP_OK) {
    if (ret == ESP_FAIL) {
      ESP_LOGE(TAG, "Failed to mount or format filesystem");
    } else if (ret == ESP_ERR_NOT_FOUND) {
      ESP_LOGE(TAG, "Failed to find SPIFFS partition");
    } else {
      ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
    }
  } else {
    ESP_LOGI(TAG, "SPIFFS mounted successfully.");
  }
}


void app_main(void) {
  // turn off the LCD backlight
  gpio_reset_pin(PIN_LCD_BL);
  gpio_set_direction(PIN_LCD_BL, GPIO_MODE_OUTPUT);
  gpio_set_level(PIN_LCD_BL, 1);

  init_spiffs();
  run_echo_loop();
}
