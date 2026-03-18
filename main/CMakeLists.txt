if(CONFIG_LLM_INFERENCE_INT8)
    set(LLM_SRC "llm8.c")
else()
    set(LLM_SRC "llm.c")
endif()

idf_component_register(SRCS "main.c" "${LLM_SRC}"
                    INCLUDE_DIRS ".")

spiffs_create_partition_image(storage ../spiffs_data FLASH_IN_PROJECT)
