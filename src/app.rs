use core::ffi::c_void;
use std::{ffi::CStr, os::raw::c_char, ptr};

use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk::{self},
    Entry,
};
pub use ash::{Device, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::windows::WindowExtWindows,
    window::{Window, WindowBuilder},
};

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(target_os = "windows")]
use winapi::um::libloaderapi::GetModuleHandleW;

use crate::utils::{self, *};

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

const REQUIRED_VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

//Todo: find a way to have one of the two, not both
const REQUIRED_DEVICE_EXTENSIONS_RAW: [*const i8; 1] =
    [ash::extensions::khr::Swapchain::name().as_ptr()];
const REQUIRED_DEVICE_EXTENSIONS: [&str; 1] = ["VK_KHR_swapchain"];

struct QueueFamiliyIndices {
    graphics_family: Option<u32>,
    present_familiy: Option<u32>,
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
    device: Option<ash::Device>,
    graphics_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,
    surface: Option<vk::SurfaceKHR>,
    surface_loader: Option<ash::extensions::khr::Surface>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_loader: Option<Swapchain>,
    debug_utils_loader: Option<DebugUtils>,
    debug_callback: Option<vk::DebugUtilsMessengerEXT>,
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
            device: None,
            graphics_queue: None,
            present_queue: None,
            surface: None,
            surface_loader: None,
            swapchain: None,
            swapchain_loader: None,
            debug_utils_loader: None,
            debug_callback: None,
        }
    }

    pub fn run(mut self) {
        let (event_loop, window) = Self::init_window();
        self.init_vulkan(&window);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => {
                    // Update renderer
                    self.render();
                    window.request_redraw();
                }
                _ => (),
            }
        });
    }

    fn init_window() -> (EventLoop<()>, Window) {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_resizable(true)
            .with_title("Vulkan Renderer")
            .build(&event_loop)
            .unwrap();
        (event_loop, window)
    }

    fn has_validation_layer_support(entry: &Entry) -> bool {
        let layer_properties = entry.enumerate_instance_layer_properties().unwrap();

        if layer_properties.is_empty() {
            eprintln!("No layers availible");
            return false;
        }

        // print availble layers
        println!("Availible Layers: ");
        for layer in layer_properties.iter() {
            println!("- {}", raw_string_to_string(&layer.layer_name));
        }

        for required_layer in REQUIRED_VALIDATION_LAYERS.iter() {
            let mut is_found = false;

            for layer in layer_properties.iter() {
                let layer_name = raw_string_to_string(&layer.layer_name);
                if *required_layer == layer_name {
                    is_found = true;
                    break;
                }
            }

            if !is_found {
                return false;
            }
        }

        true
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        unsafe {
            let app_name = CStr::from_bytes_with_nul_unchecked(b"VulkanTriangle\0");
            let engine_name = CStr::from_bytes_with_nul_unchecked(b"VulkanRenderer\0");

            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: 0,
                p_engine_name: engine_name.as_ptr(),
                engine_version: 0,
                api_version: vk::make_api_version(0, 1, 0, 0),
                ..Default::default()
            };

            let mut extension_names = ash_window::enumerate_required_extensions(window)
                .unwrap()
                .to_vec();
            extension_names.push(DebugUtils::name().as_ptr());
            extension_names
                .push(CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_utils\0").as_ptr());

            let mut layer_names: Vec<&CStr> = Vec::new();
            if VALIDATION_LAYERS_ENABLED && Self::has_validation_layer_support(entry) {
                println!("VALIDATION LAYERS ACTIVE");
                layer_names.push(CStr::from_bytes_with_nul_unchecked(
                    b"VK_LAYER_KHRONOS_validation\0",
                ));
            }

            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                pp_enabled_layer_names: layers_names_raw.as_ptr(),
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_layer_count: layer_names.len() as u32,
                enabled_extension_count: extension_names.len() as u32,
                flags: vk::InstanceCreateFlags::default(),
                ..Default::default()
            };

            entry.create_instance(&create_info, None).unwrap()
        }
    }

    fn init_debug_messenger(&mut self) {
        if !VALIDATION_LAYERS_ENABLED {
            return;
        }
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(vulkan_debug_callback),
            p_user_data: ptr::null_mut(),
        };

        assert!(self.entry.is_some());
        assert!(self.instance.is_some());

        self.debug_utils_loader = Some(DebugUtils::new(
            self.entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        ));

        assert!(self.debug_utils_loader.is_some());

        self.debug_callback = unsafe {
            Some(
                self.debug_utils_loader
                    .as_ref()
                    .unwrap()
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap(),
            )
        };
    }

    #[cfg(target_os = "windows")]
    fn create_surface(
        &mut self,
        window: &Window,
    ) -> (vk::SurfaceKHR, ash::extensions::khr::Surface) {
        // TODO: Windows only atm. Add support for other platforms later.
        unsafe {
            let hwnd = window.hwnd();
            let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
                s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                hinstance,
                hwnd: hwnd as *const c_void,
            };
            let win32_surface_loader = Win32Surface::new(
                self.entry.as_ref().unwrap(),
                self.instance.as_ref().unwrap(),
            );
            (
                win32_surface_loader
                    .create_win32_surface(&win32_create_info, None)
                    .unwrap(),
                ash::extensions::khr::Surface::new(
                    self.entry.as_ref().unwrap(),
                    self.instance.as_ref().unwrap(),
                ),
            )
        }
    }

    fn get_physical_device(&mut self) -> Option<vk::PhysicalDevice> {
        let devices = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices()
                .unwrap()
        };

        println!("Found {} devices with Vulkan support.", devices.len());
        let mut best_device: Option<vk::PhysicalDevice> = None;

        for device in devices {
            let instance = self.instance.as_ref().unwrap();
            if self.is_physical_device_suitable(instance, &device) {
                let device_properties = unsafe { instance.get_physical_device_properties(device) };
                println!(
                    "{:?} is suitable.",
                    raw_string_to_string(&device_properties.device_name)
                );
                best_device = Some(device);
            }
        }

        best_device
    }

    fn is_physical_device_suitable(
        &self,
        instance: &Instance,
        device: &vk::PhysicalDevice,
    ) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
        let device_features = unsafe { instance.get_physical_device_features(*device) };

        let indices: QueueFamiliyIndices = Self::find_queue_families(
            instance,
            device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let extensions_supported = Self::check_device_extension_support(instance, device);

        let mut swap_chain_supported = false;
        if extensions_supported {
            let swap_chain_support = Self::query_swapchain_support(
                device,
                self.surface.as_ref().unwrap(),
                self.surface_loader.as_ref().unwrap(),
            );

            swap_chain_supported = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty()
        }

        device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            && device_features.geometry_shader > 0
            && indices.graphics_family.is_some()
            && extensions_supported
            && swap_chain_supported
    }

    fn find_queue_families(
        instance: &Instance,
        device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> QueueFamiliyIndices {
        let mut indices = QueueFamiliyIndices {
            graphics_family: None,
            present_familiy: None,
        };

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        for (index, family) in (0_u32..).zip(queue_family_properties.iter()) {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(index);
            }

            let is_present_supported = unsafe {
                surface_loader
                    .get_physical_device_surface_support(*device, index, *surface)
                    .unwrap()
            };

            if family.queue_count > 0 && is_present_supported {
                indices.present_familiy = Some(index);
            }

            if !family.queue_flags.is_empty() {
                break;
            }
        }
        indices
    }

    fn check_device_extension_support(instance: &Instance, device: &vk::PhysicalDevice) -> bool {
        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .expect("Error while getting device extension properties")
        };

        let mut found_extensions = vec![];
        for extension in extension_properties.iter() {
            let extension_name = utils::raw_string_to_string(&extension.extension_name);

            if REQUIRED_DEVICE_EXTENSIONS.contains(&extension_name.as_str()) {
                found_extensions.push(extension_name);
            }
        }

        println!("Found required extensions({}):", found_extensions.len());
        for found_names in found_extensions.iter() {
            println!("- {}", found_names);
        }

        found_extensions.len() == REQUIRED_DEVICE_EXTENSIONS.len()
    }

    fn query_swapchain_support(
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> SwapChainSupportDetails {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .expect("Failed to query for surface capabilities.");
            let formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .expect("Failed to query for surface formats.");
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn choose_swap_surface_format(
        available_formats: Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats.iter() {
            if format.format == vk::Format::B8G8R8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(
        available_present_modes: Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for present_mode in available_present_modes.iter() {
            if *present_mode == vk::PresentModeKHR::MAILBOX {
                return vk::PresentModeKHR::MAILBOX;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: clamp(
                    WIDTH as u32,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    HEIGHT as u32,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_swap_chain(&mut self, physical_device: &vk::PhysicalDevice) {
        let swap_chain_support = Self::query_swapchain_support(
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let surface_format = Self::choose_swap_surface_format(swap_chain_support.formats);
        let present_mode = Self::choose_swap_present_mode(swap_chain_support.present_modes);
        let extent = Self::choose_swap_extent(swap_chain_support.capabilities);

        let image_count = swap_chain_support.capabilities.min_image_count + 1;
        let image_count = if swap_chain_support.capabilities.max_image_count > 0 {
            image_count.min(swap_chain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: *self.surface.as_ref().unwrap(),
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let indices = Self::find_queue_families(
            self.instance.as_ref().unwrap(),
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );
        let queue_family_indices = vec![
            indices.graphics_family.unwrap(),
            indices.present_familiy.unwrap(),
        ];

        if indices.graphics_family != indices.present_familiy {
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        } else {
            create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = ptr::null();
        }

        self.swapchain_loader = Some(ash::extensions::khr::Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
        ));
        self.swapchain = unsafe {
            Some(
                self.swapchain_loader
                    .as_ref()
                    .unwrap()
                    .create_swapchain(&create_info, None)
                    .expect("Failed to create Swapchain!"),
            )
        };
    }

    fn create_logical_device(&mut self, physical_device: &vk::PhysicalDevice) {
        let indices = Self::find_queue_families(
            self.instance.as_ref().unwrap(),
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let physical_device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let requred_validation_layer_raw_names: Vec<&CStr> = REQUIRED_VALIDATION_LAYERS
            .iter()
            .map(|layer_name| unsafe { CStr::from_bytes_with_nul_unchecked(layer_name.as_bytes()) })
            .collect();

        let enable_layer_names: Vec<*const c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: &queue_create_info,
            queue_create_info_count: 1,
            p_enabled_features: &physical_device_features,
            enabled_extension_count: REQUIRED_DEVICE_EXTENSIONS_RAW.len() as u32,
            pp_enabled_extension_names: REQUIRED_DEVICE_EXTENSIONS_RAW.as_ptr(),
            enabled_layer_count: if VALIDATION_LAYERS_ENABLED {
                REQUIRED_VALIDATION_LAYERS.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names: if VALIDATION_LAYERS_ENABLED {
                enable_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            ..Default::default()
        };

        let device = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .create_device(*physical_device, &create_info, None)
        };
        self.device = Some(device.expect("Error while creating logical device."));

        self.graphics_queue = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.graphics_family.unwrap(), 0)
        });

        self.present_queue = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.present_familiy.unwrap(), 0)
        });
    }

    fn init_vulkan(&mut self, window: &Window) {
        self.entry = Some(Entry::linked());
        self.instance = Some(Self::create_instance(self.entry.as_ref().unwrap(), window));
        self.init_debug_messenger();
        let (surface, surface_loader) = self.create_surface(window);
        self.surface = Some(surface);
        self.surface_loader = Some(surface_loader);

        let physical_device = self
            .get_physical_device()
            .expect("Error while getting physical device");
        self.create_logical_device(&physical_device);
        self.create_swap_chain(&physical_device);
    }

    fn render(&self) {}
    pub fn shutdown(&self) {
        println!("Shutdown called");
        unsafe {
            self.device.as_ref().unwrap().destroy_device(None);
            self.surface_loader
                .as_ref()
                .unwrap()
                .destroy_surface(*self.surface.as_ref().unwrap(), None);

            if VALIDATION_LAYERS_ENABLED {
                self.debug_utils_loader
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(*self.debug_callback.as_ref().unwrap(), None);
            }

            self.instance.as_ref().unwrap().destroy_instance(None);
        }
        println!("Shutdown okay");
    }
}

fn clamp(value: u32, min: u32, max: u32) -> u32 {
    let mut value = value;
    if value < min {
        value = min
    }
    if value > max {
        value = max
    }
    value
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.shutdown();
    }
}
