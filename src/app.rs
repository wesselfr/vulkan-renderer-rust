use std::{ffi::CStr, os::raw::c_char, ptr};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self},
    Entry,
};
pub use ash::{Device, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::utils::*;

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

const REQUIRED_VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

struct QueueFamiliyIndices {
    graphics_family: Option<u32>,
}

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
    device: Option<ash::Device>,
    debug_utils_loader: Option<DebugUtils>,
    debug_callback: Option<vk::DebugUtilsMessengerEXT>,
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
            device: None,
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

    // Todo: add checks for validaiton layers.
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

        let indices: QueueFamiliyIndices = Self::find_queue_families(instance, device);

        device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            && device_features.geometry_shader > 0
            && indices.graphics_family.is_some()
    }

    fn find_queue_families(
        instance: &Instance,
        device: &vk::PhysicalDevice,
    ) -> QueueFamiliyIndices {
        let mut indices = QueueFamiliyIndices {
            graphics_family: None,
        };

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        for (index, family) in (0_u32..).zip(queue_family_properties.iter()) {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(index);
                break;
            }
        }
        indices
    }

    fn create_logical_device(&mut self, physical_device: &vk::PhysicalDevice) {
        let indices = Self::find_queue_families(self.instance.as_ref().unwrap(), physical_device);

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
            enabled_extension_count: 0,
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

        let graphics_queue = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.graphics_family.unwrap(), 0)
        };
    }

    fn init_vulkan(&mut self, window: &Window) {
        self.entry = Some(Entry::linked());
        self.instance = Some(Self::create_instance(self.entry.as_ref().unwrap(), window));
        self.init_debug_messenger();
        let physical_device = self
            .get_physical_device()
            .expect("Error while getting physical device");
        self.create_logical_device(&physical_device);
    }

    fn render(&self) {}
    pub fn shutdown(&self) {
        println!("Shutdown called");
        unsafe {
            self.device.as_ref().unwrap().destroy_device(None);

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
