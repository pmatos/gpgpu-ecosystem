#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <iostream>
#include <renderdoc_app.h>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "shaders/vulkan.h"

#define STR(x) #x

const std::vector<const char *> enabledExtensions = {
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    // other extensions...
};

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
    };

std::vector<char> readShaderCode(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

VkShaderModule createShaderModule(const std::vector<char> &code,
                                  VkDevice &device) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shaderModule;
}

// Function to find a suitable memory type
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type.");
}

struct BufferMemory {
  VkBuffer buffer;
  VkDeviceMemory memory;
};

BufferMemory createBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                          VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer;
  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice, memRequirements.memoryTypeBits, properties);

  VkDeviceMemory bufferMemory;
  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(device, buffer, bufferMemory, 0);

  BufferMemory bufferMemoryStruct;
  bufferMemoryStruct.buffer = buffer;
  bufferMemoryStruct.memory = bufferMemory;

  return bufferMemoryStruct;
}

void copyDataToBuffer(VkDevice device, BufferMemory bufferMemory, float *data,
                      VkDeviceSize size) {
  float *mappedMemory = nullptr;
  if (vkMapMemory(device, bufferMemory.memory, 0, size, 0,
                  (void **)&mappedMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to map memory!");
  }
  // could do memcpy here
  for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
    mappedMemory[i] = data[i];
  }
  vkUnmapMemory(device, bufferMemory.memory);
}

void copyDataFromBuffer(VkDevice device, BufferMemory bufferMemory, void *data,
                        VkDeviceSize size) {
  void *mappedMemory;
  vkMapMemory(device, bufferMemory.memory, 0, size, 0, &mappedMemory);
  memcpy(data, mappedMemory, (size_t)size);
  vkUnmapMemory(device, bufferMemory.memory);
}

// Returns the first device from the vector of devices that's an
// Nvidia GPU, or null otherwise
VkPhysicalDevice chooseNvidiaDevice(std::vector<VkPhysicalDevice> &physds) {
  for (auto physd : physds) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physd, &deviceProperties);

    if (deviceProperties.vendorID ==
        0x10DE) { // 0x10DE is the vendor ID for Nvidia.
      // Lets ensure we can use vulkan 1.2
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(physd, &deviceProperties);

      assert(deviceProperties.apiVersion >= VK_API_VERSION_1_2);
      return physd;
    }
  }

  return VK_NULL_HANDLE;
}

bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

bool IsExtensionSupported(const char *extensionName) {
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                         availableExtensions.data());

  for (const auto &extension : availableExtensions) {
    if (strcmp(extension.extensionName, extensionName) == 0) {
      return true;
    }
  }

  return false;
}

void vkDestroyBufferMemory(VkDevice device, BufferMemory bufferMemory) {
  vkDestroyBuffer(device, bufferMemory.buffer, nullptr);
  vkFreeMemory(device, bufferMemory.memory, nullptr);
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType [[maybe_unused]],
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData [[maybe_unused]]) {
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cout << "Warning: " << pCallbackData->messageIdNumber << ":"
              << pCallbackData->pMessageIdName << ":" << pCallbackData->pMessage
              << std::endl;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cerr << "Error: " << pCallbackData->messageIdNumber << ":"
              << pCallbackData->pMessageIdName << ":" << pCallbackData->pMessage
              << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    std::cout << "Verbose: " << pCallbackData->messageIdNumber << ":"
              << pCallbackData->pMessageIdName << ":" << pCallbackData->pMessage
              << std::endl;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cout << "Info: " << pCallbackData->messageIdNumber << ":"
              << pCallbackData->pMessageIdName << ":" << pCallbackData->pMessage
              << std::endl;
  }
  return VK_FALSE;
}
int main() {

  // 1. Create Vulkan instance and device with compute queue

  // Setup debug callback
  // Declare a function pointer with the same signature as
  // vkCreateDebugUtilsMessengerEXT
  if (!IsExtensionSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    throw std::runtime_error("VK_EXT_debug_utils extension not available!");
  }

  VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = {};
  debugUtilsCreateInfo.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debugUtilsCreateInfo.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  debugUtilsCreateInfo.messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debugUtilsCreateInfo.pfnUserCallback = debugCallback;
  debugUtilsCreateInfo.pUserData = nullptr;

  // 1.1. Create Vulkan instance
  VkValidationFeaturesEXT validationFeatures = {};
  validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
  validationFeatures.enabledValidationFeatureCount = 1;

  VkValidationFeatureEnableEXT enabledValidationFeatures[1] = {
      VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

  VkApplicationInfo appInfo = {
      .apiVersion = VK_API_VERSION_1_2,
  };

  VkInstance instance;
  VkInstanceCreateInfo instanceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = &debugUtilsCreateInfo,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
      .ppEnabledLayerNames = validationLayers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
      .ppEnabledExtensionNames = enabledExtensions.data()};
  validationFeatures.pNext = instanceCreateInfo.pNext;
  instanceCreateInfo.pNext = &validationFeatures;
  vkCreateInstance(&instanceCreateInfo, NULL, &instance);

  // Load the functions
  PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
  vkCreateDebugUtilsMessengerEXT =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkCreateDebugUtilsMessengerEXT");
  if (vkCreateDebugUtilsMessengerEXT == nullptr) {
    throw std::runtime_error("Failed to load vkCreateDebugUtilsMessengerEXT");
  }
  PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
  vkDestroyDebugUtilsMessengerEXT =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkDestroyDebugUtilsMessengerEXT");
  if (vkCreateDebugUtilsMessengerEXT == nullptr) {
    throw std::runtime_error("Failed to load vkDestroyDebugUtilsMessengerEXT");
  }

  VkDebugUtilsMessengerEXT debugMessenger;
  if (vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsCreateInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
    throw std::runtime_error("Failed to set up debug messenger!");
  }

  // 1.2. Create physical device, device and get device queue
  VkQueue queue;
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    std::cerr << "failed to find GPUs with Vulkan support!\n";
    exit(-1);
  }
  std::cout << "Found " << deviceCount << " devices with Vulkan support\n";

  // Choose physical device.
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  // Need to choose one of the Physical devices
  VkPhysicalDevice physicalDevice = chooseNvidiaDevice(devices);
  if (physicalDevice == VK_NULL_HANDLE) {
    std::cerr << "failed to find a suitable nvidia GPU!\n";
    return -1;
  }

  // Create logical device and get compute queue
  VkDevice device;
  const std::vector<const char *> deviceExtensions = {
      VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME};
  const float queuePrio = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &queuePrio};
  VkDeviceCreateInfo deviceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = NULL,
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data(),
      .pEnabledFeatures = NULL};
  vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
  vkGetDeviceQueue(device, 0, 0, &queue);

  // 2. Create GPU buffers for matrices and allocate memory
  // and copy host matrices to GPU buffers
  const size_t matrixSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
  float hostA[MATRIX_SIZE * MATRIX_SIZE];
  float hostB[MATRIX_SIZE * MATRIX_SIZE];
  float hostC[MATRIX_SIZE * MATRIX_SIZE];

  // hostA is the identify matrix, hostB is a matrix with increasing values
  // therefore hostC will end up being equal to hostB after the multiplication.
  // For now, we initialize it with zeros.
  for (int row = 0; row < MATRIX_SIZE; row++) {
    for (int col = 0; col < MATRIX_SIZE; col++) {
      hostA[row * MATRIX_SIZE + col] = row == col ? 1.0f : 0.0f;
    }
  }
  for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
    hostB[i] = static_cast<float>(i);
  }
  for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
    hostC[i] = 0.0f;
  }

  BufferMemory bufferMemoryA = createBuffer(
      device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  BufferMemory bufferMemoryB = createBuffer(
      device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  BufferMemory bufferMemoryC = createBuffer(
      device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  copyDataToBuffer(device, bufferMemoryA, hostA, matrixSize);
  copyDataToBuffer(device, bufferMemoryB, hostB, matrixSize);

  // 4. Create descriptor set layout, pool and set
  VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[] = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
       .pImmutableSamplers = nullptr},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
       .pImmutableSamplers = nullptr},
      {.binding = 2,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
       .pImmutableSamplers = nullptr}};
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .bindingCount = 3,
      .pBindings = descriptorSetLayoutBinding};
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL,
                                  &descriptorSetLayout) != VK_SUCCESS) {
    std::cerr << "failed to create descriptor set layout!\n";
    exit(-1);
  }

  VkDescriptorPoolSize descriptorPoolSize = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3};
  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &descriptorPoolSize};
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL,
                             &descriptorPool) != VK_SUCCESS) {
    std::cerr << "failed to create descriptor pool!\n";
    exit(-1);
  }

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = NULL,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptorSetLayout};
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                               &descriptorSet) != VK_SUCCESS) {
    std::cerr << "failed to allocate descriptor set!\n";
    exit(-1);
  }

  VkDescriptorBufferInfo descriptorBufferInfoA = {
      .buffer = bufferMemoryA.buffer, .offset = 0, .range = matrixSize};
  VkDescriptorBufferInfo descriptorBufferInfoB = {
      .buffer = bufferMemoryB.buffer, .offset = 0, .range = matrixSize};
  VkDescriptorBufferInfo descriptorBufferInfoC = {
      .buffer = bufferMemoryC.buffer, .offset = 0, .range = matrixSize};
  VkWriteDescriptorSet writeDescriptorSet[] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .pNext = NULL,
       .dstSet = descriptorSet,
       .dstBinding = 0,
       .dstArrayElement = 0,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pImageInfo = NULL,
       .pBufferInfo = &descriptorBufferInfoA,
       .pTexelBufferView = NULL},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .pNext = NULL,
       .dstSet = descriptorSet,
       .dstBinding = 1,
       .dstArrayElement = 0,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pImageInfo = NULL,
       .pBufferInfo = &descriptorBufferInfoB,
       .pTexelBufferView = NULL},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .pNext = NULL,
       .dstSet = descriptorSet,
       .dstBinding = 2,
       .dstArrayElement = 0,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pImageInfo = NULL,
       .pBufferInfo = &descriptorBufferInfoC,
       .pTexelBufferView = NULL}};
  vkUpdateDescriptorSets(device, 3, writeDescriptorSet, 0, NULL);

  // 5. Create pipeline, shader module and pipeline layout
  auto shaderCode = readShaderCode("src/shaders/matrixmul.spv");
  VkShaderModule shaderModule = createShaderModule(shaderCode, device);

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout};
  VkPipelineLayout pipelineLayout;
  if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL,
                             &pipelineLayout) != VK_SUCCESS) {
    std::cerr << "failed to create pipeline layout!\n";
    exit(-1);
  }

  VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shaderModule,
      .pName = "main"};
  VkComputePipelineCreateInfo computePipelineCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = pipelineShaderStageCreateInfo,
      .layout = pipelineLayout};
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkResult CreateCP =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                               &computePipelineCreateInfo, nullptr, &pipeline);
  if (CreateCP != VK_SUCCESS && CreateCP != VK_PIPELINE_COMPILE_REQUIRED_EXT) {
    if (CreateCP == VK_ERROR_OUT_OF_HOST_MEMORY)
      std::cerr << "Out of host memory\n";
    else if (CreateCP == VK_ERROR_OUT_OF_DEVICE_MEMORY)
      std::cerr << "Out of device memory\n";
    else if (CreateCP == VK_ERROR_INVALID_SHADER_NV)
      std::cerr << "Invalid shader\n";
    std::cerr << "Error: Create compute pipeline failed\n";
    exit(-1);
  }

  // 6. Create command pool, command buffer and execute command buffer
  VkCommandPoolCreateInfo commandPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .queueFamilyIndex = 0};
  VkCommandPool commandPool = VK_NULL_HANDLE;
  if (vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool) !=
      VK_SUCCESS) {
    std::cerr << "failed to create command pool!\n";
    exit(-1);
  }

  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = NULL,
      .commandPool = commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                               &commandBuffer) != VK_SUCCESS) {
    std::cerr << "failed to allocate command buffers!\n";
    exit(-1);
  }

  VkCommandBufferBeginInfo commandBufferBeginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = NULL,
      .flags = 0,
      .pInheritanceInfo = NULL};
  if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) !=
      VK_SUCCESS) {
    std::cerr << "failed to begin recording command buffer!\n";
    exit(-1);
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

  // Capture with renderdoc
  RENDERDOC_API_1_6_0 *rdoc_api = NULL;
  pRENDERDOC_GetAPI RENDERDOC_GetAPI = NULL;
  void *mod = dlopen("/usr/lib/librenderdoc.so", RTLD_NOW);
  if (mod) {
    RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
  } else
    throw std::runtime_error("Failed to load librenderdoc.so");

  // Now, get the RenderDoc API
  int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void **)&rdoc_api);
  if (ret != 1) {
    // Failed to get RenderDoc API
    throw std::runtime_error("Failed to get RenderDoc API");
  }

  // Start RenderDoc frame capture
  rdoc_api->StartFrameCapture(device, nullptr);

  vkCmdDispatch(commandBuffer, MATRIX_SIZE, MATRIX_SIZE, 1);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    std::cerr << "failed to record command buffer!\n";
    exit(-1);
  }

  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = NULL,
                             .waitSemaphoreCount = 0,
                             .pWaitSemaphores = NULL,
                             .pWaitDstStageMask = NULL,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commandBuffer,
                             .signalSemaphoreCount = 0,
                             .pSignalSemaphores = NULL};
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
    std::cerr << "failed to submit draw command buffer!\n";
    exit(-1);
  }
  if (vkQueueWaitIdle(queue) != VK_SUCCESS) {
    std::cerr << "failed to wait idle!\n";
    exit(-1);
  }

  // End RenderDoc frame capture
  rdoc_api->EndFrameCapture(device, nullptr);
  dlclose(mod);

  // 7. Copy result matrix from GPU buffer to host
  copyDataFromBuffer(device, bufferMemoryC, hostC, matrixSize);

  // 8. Verify result matrix
#if 0
  bool isZero = true;
  for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
    if (fabs(hostC[i]) > 0.000001f) {
      isZero = false;
    }
  }
  if (isZero) {
    std::cerr << "Error: result matrix is zero\n";
  }
#endif
  float verMatrix[MATRIX_SIZE * MATRIX_SIZE];
  for (int row = 0; row < MATRIX_SIZE; row++) {
    for (int col = 0; col < MATRIX_SIZE; col++) {
      float sum = 0.0f;
      for (int i = 0; i < MATRIX_SIZE; i++) {
        sum += hostA[row * MATRIX_SIZE + i] * hostB[i * MATRIX_SIZE + col];
      }
      verMatrix[row * MATRIX_SIZE + col] = sum;
    }
  }

  // Print verMatrix
  std::cout << "verMatrix:\n";
  for (int row = 0; row < MATRIX_SIZE; row++) {
    for (int col = 0; col < MATRIX_SIZE; col++) {
      std::cout << verMatrix[row * MATRIX_SIZE + col] << " ";
    }
    std::cout << "\n";
  }

  bool error = false;
  for (int row = 0; row < MATRIX_SIZE; row++) {
    for (int col = 0; col < MATRIX_SIZE; col++) {
      if (fabsf(hostC[row * MATRIX_SIZE + col] -
                verMatrix[row * MATRIX_SIZE + col]) > 0.000001f) {
        printf("Verification failed: (%d,%d) expected %f but got %f\n", row,
               col, verMatrix[row * MATRIX_SIZE + col],
               hostC[row * MATRIX_SIZE + col]);
        error = true;
      }
      if (error)
        break;
    }
    if (error)
      break;
  }

  // 9. Release resources
  vkDestroyShaderModule(device, shaderModule, NULL);
  vkDestroyPipeline(device, pipeline, NULL);
  vkDestroyPipelineLayout(device, pipelineLayout, NULL);
  vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
  vkDestroyDescriptorPool(device, descriptorPool, NULL);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
  vkDestroyBufferMemory(device, bufferMemoryA);
  vkDestroyBufferMemory(device, bufferMemoryB);
  vkDestroyBufferMemory(device, bufferMemoryC);
  vkDestroyCommandPool(device, commandPool, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  vkDestroyInstance(instance, NULL);

  return 0;
}