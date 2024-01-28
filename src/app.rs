use core::ffi::c_void;
use std::{
    ffi::{CStr, CString},
    mem::size_of,
    os::raw::c_char,
    path::{self, Path},
    ptr,
    time::Instant,
};

use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk::{
        self, ColorComponentFlags, CullModeFlags, FrontFace, PolygonMode, PrimitiveTopology,
        SampleCountFlags, ShaderStageFlags,
    },
    Entry,
};
pub use ash::{Device, Instance};
use glam::{Vec2, Vec3};
use image::{EncodableLayout, GenericImageView};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::windows::WindowExtWindows,
    window::{Window, WindowBuilder},
};

use memoffset::offset_of;

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(target_os = "windows")]
use winapi::um::libloaderapi::GetModuleHandleW;

use crate::utils::{self, *};

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

const TEXTURE_PATH: &'static str = "assets/textures/texture.jpg";

const MAX_FRAMES_IN_FLIGHT: usize = 2;

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

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}

const VERTICES: [Vertex; 4] = [
    Vertex {
        pos: Vec2 { x: -0.5, y: -0.5 },
        color: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    },
    Vertex {
        pos: Vec2 { x: 0.5, y: -0.5 },
        color: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    },
    Vertex {
        pos: Vec2 { x: 0.5, y: 0.5 },
        color: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    },
    Vertex {
        pos: Vec2 { x: -0.5, y: 0.5 },
        color: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    },
];

const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

struct UniformBufferObject {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
    device: Option<ash::Device>,
    physical_device: Option<vk::PhysicalDevice>,
    graphics_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,
    surface: Option<vk::SurfaceKHR>,
    surface_loader: Option<ash::extensions::khr::Surface>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_loader: Option<Swapchain>,
    swapchain_images: Option<Vec<vk::Image>>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    swapchain_format: Option<vk::Format>,
    swapchain_extent: Option<vk::Extent2D>,
    swapchain_frame_buffers: Option<Vec<vk::Framebuffer>>,
    render_pass: Option<vk::RenderPass>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pipeline_layout: Option<vk::PipelineLayout>,
    graphics_pipeline: Option<vk::Pipeline>,
    command_pool: Option<vk::CommandPool>,
    command_buffers: Option<Vec<vk::CommandBuffer>>,
    image_available_semaphores: Option<Vec<vk::Semaphore>>,
    render_finished_semaphores: Option<Vec<vk::Semaphore>>,
    in_flight_fences: Option<Vec<vk::Fence>>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    uniform_buffers: Option<Vec<Buffer>>,
    uniform_buffers_mapped: Option<Vec<vk::DeviceMemory>>,
    image_textures: Option<Vec<(vk::Image, vk::DeviceMemory)>>,
    texture_image_view: Option<vk::ImageView>,
    texture_sampler: Option<vk::Sampler>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Option<Vec<vk::DescriptorSet>>,
    debug_utils_loader: Option<DebugUtils>,
    debug_callback: Option<vk::DebugUtilsMessengerEXT>,
    frame_index: usize,
    delta_time: f32,
    time: f32,
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
            device: None,
            physical_device: None,
            graphics_queue: None,
            present_queue: None,
            surface: None,
            surface_loader: None,
            swapchain: None,
            swapchain_loader: None,
            swapchain_images: None,
            swapchain_image_views: None,
            swapchain_format: None,
            swapchain_extent: None,
            swapchain_frame_buffers: None,
            render_pass: None,
            descriptor_set_layout: None,
            pipeline_layout: None,
            graphics_pipeline: None,
            command_pool: None,
            command_buffers: None,
            image_available_semaphores: None,
            render_finished_semaphores: None,
            in_flight_fences: None,
            vertex_buffer: None,
            index_buffer: None,
            uniform_buffers: None,
            uniform_buffers_mapped: None,
            image_textures: None,
            texture_image_view: None,
            texture_sampler: None,
            descriptor_pool: None,
            descriptor_sets: None,
            debug_utils_loader: None,
            debug_callback: None,
            frame_index: 0,
            delta_time: 0.0,
            time: 0.0,
        }
    }

    pub fn run(mut self) {
        let (event_loop, window) = Self::init_window();
        self.init_vulkan(&window);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => {
                    // We don't render when minimized
                    if window.inner_size().width != 0 && window.inner_size().height != 0 {
                        let now = Instant::now();

                        // Update renderer
                        self.render();
                        window.request_redraw();

                        self.delta_time = now.elapsed().as_secs_f32();
                    }
                }
                _ => {}
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

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let memory_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_memory_properties(*self.physical_device.as_ref().unwrap())
        };

        for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(properties) {
                return i as u32;
            }
        }
        panic!("Failed to find suitable memory type!");
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

        let surface_format: vk::SurfaceFormatKHR =
            Self::choose_swap_surface_format(swap_chain_support.formats);
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
        let queue_family_indices = [
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

        let present_images = unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .get_swapchain_images(*self.swapchain.as_ref().unwrap())
                .unwrap()
        };

        self.swapchain_images = Some(Vec::new());
        self.swapchain_images
            .as_mut()
            .unwrap()
            .clone_from(&present_images);

        self.swapchain_extent = Some(extent);
        self.swapchain_format = Some(surface_format.format);
    }

    fn create_image_view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&view_info, None)
                .expect("Failed to create image view!")
        };

        image_view
    }

    fn create_swapchain_image_views(&mut self) {
        self.swapchain_image_views = Some(Vec::new());
        self.swapchain_image_views
            .as_mut()
            .unwrap()
            .reserve(self.swapchain_images.as_ref().unwrap().len());

        for image in self.swapchain_images.as_ref().unwrap() {
            let image_view = self.create_image_view(*image, self.swapchain_format.unwrap());
            self.swapchain_image_views
                .as_mut()
                .unwrap()
                .push(image_view);
        }
    }

    fn create_shader_module(&mut self, code: &Vec<u8>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_shader_module(&create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn create_render_pass(&mut self) {
        let color_attachment = vk::AttachmentDescription {
            format: *self.swapchain_format.as_ref().unwrap(),
            samples: SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let dependency = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let render_pass_attachments = [color_attachment];

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: dependency.as_ptr(),
            ..Default::default()
        };

        self.render_pass = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_render_pass(&render_pass_info, None)
                    .expect("Failed to create render pass!"),
            )
        }
    }

    fn create_graphics_pipeline(&mut self) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_code = read_shader_file(Path::new("assets/shaders/vert.spv"))
            .expect("Error while reading vertex shader");
        let fragment_shader_code = read_shader_file(Path::new("assets/shaders/frag.spv"))
            .expect("Error while reading fragment shader");

        let vertex_shader_module = self.create_shader_module(&vertex_shader_code);
        let fragment_shader_module = self.create_shader_module(&fragment_shader_code);

        let main_function = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::VERTEX,
                module: vertex_shader_module,
                p_name: main_function.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::FRAGMENT,
                module: fragment_shader_module,
                p_name: main_function.as_ptr(),
                ..Default::default()
            },
        ];

        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_binding_descriptions: &binding_description,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.as_ref().unwrap().width as f32,
            height: self.swapchain_extent.as_ref().unwrap().height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *self.swapchain_extent.as_ref().unwrap(),
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            ..Default::default()
        };

        let rasterizer_state_create_info = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            ..Default::default()
        }];

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: color_blend_attachments.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: self.descriptor_set_layout.as_ref().unwrap(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let pipeline_dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterizer_state_create_info,
            //p_multisample_state: TODO
            p_color_blend_state: &color_blending,
            p_dynamic_state: &pipeline_dynamic_state_info,
            layout: pipeline_layout,
            render_pass: *self.render_pass.as_ref().unwrap(),
            subpass: 0,
            ..Default::default()
        }];

        let graphics_pipelines = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .expect("Failed to create graphics pipeline.")
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(vertex_shader_module, None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(fragment_shader_module, None);
        }

        (graphics_pipelines[0], pipeline_layout)
    }

    fn create_descriptor_set_layout(&mut self) {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        };

        let bindings = [ubo_layout_binding];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        self.descriptor_set_layout = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_set_layout(&layout_info, None)
                    .expect("Failed to create descriptor set layout!"),
            )
        }
    }

    fn create_frame_buffers(&mut self) {
        self.swapchain_frame_buffers = Some(Vec::new());
        self.swapchain_frame_buffers
            .as_mut()
            .unwrap()
            .reserve(self.swapchain_image_views.as_ref().unwrap().len());

        for swap_chain_image_view in self.swapchain_image_views.as_ref().unwrap() {
            let framebuffer_info = vk::FramebufferCreateInfo {
                render_pass: *self.render_pass.as_ref().unwrap(),
                attachment_count: 1,
                p_attachments: swap_chain_image_view,
                width: self.swapchain_extent.as_ref().unwrap().width,
                height: self.swapchain_extent.as_ref().unwrap().height,
                layers: 1,
                ..Default::default()
            };

            self.swapchain_frame_buffers.as_mut().unwrap().push(unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_framebuffer(&framebuffer_info, None)
                    .expect("Failed to create framebuffer!")
            })
        }
    }

    fn create_logical_device(
        &mut self,
        physical_device: &vk::PhysicalDevice,
    ) -> QueueFamiliyIndices {
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

        indices
    }

    fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: *self.command_pool.as_ref().unwrap(),
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer = unsafe {
            *self
                .device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate command buffers")
                .first()
                .unwrap()
        };

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer.");
        }

        command_buffer
    }

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            let device = self.device.as_ref().unwrap();
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");

            let command_buffers_for_submit = [command_buffer];

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: command_buffers_for_submit.as_ptr(),
                ..Default::default()
            };
            device
                .queue_submit(
                    *self.graphics_queue.as_ref().unwrap(),
                    &[submit_info],
                    vk::Fence::null(),
                )
                .expect("Failed to submit queue");
            device
                .queue_wait_idle(*self.graphics_queue.as_ref().unwrap())
                .unwrap();

            device.free_command_buffers(
                *self.command_pool.as_ref().unwrap(),
                &command_buffers_for_submit,
            );
        }
    }

    /// Creates a vkBuffer and allocates device memory for it.
    /// Todo: vkAllocate calls are limited and discouraged, improve memory management later!
    fn create_buffer(
        &mut self,
        create_info: vk::BufferCreateInfo,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        let buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_buffer(&create_info, None)
                .expect("Failed to create buffer!")
        };

        let memory_requirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_buffer_memory_requirements(buffer)
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: memory_requirements.size,
            memory_type_index: self
                .find_memory_type(memory_requirements.memory_type_bits, properties),
            ..Default::default()
        };

        let memory = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory!")
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind buffer memory!");
        }

        Buffer { buffer, memory }
    }

    fn copy_buffer(&self, src: &Buffer, dst: &Buffer, size: vk::DeviceSize) -> Result<(), String> {
        let command_buffer = self.begin_single_time_commands();

        let copy_region = vk::BufferCopy {
            size,
            ..Default::default()
        };

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                src.buffer,
                dst.buffer,
                &[copy_region],
            );
        }

        self.end_single_time_commands(command_buffer);

        Ok(())
    }

    fn destroy_buffer(&self, buffer: &Buffer) {
        let device: &Device = self.device.as_ref().unwrap();
        unsafe {
            device.destroy_buffer(buffer.buffer, None);
            device.free_memory(buffer.memory, None);
        }
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier {
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition!");
        }

        unsafe {
            let device = self.device.as_ref().unwrap();
            device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        }

        self.end_single_time_commands(command_buffer);
    }

    fn copy_buffer_to_image(&self, buffer: &Buffer, image: vk::Image, width: u32, height: u32) {
        let command_buffer = self.begin_single_time_commands();

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,

            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
                ..Default::default()
            },

            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                command_buffer,
                buffer.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        }

        self.end_single_time_commands(command_buffer);
    }

    fn create_texture_image(&mut self, path: &Path) -> (vk::Image, vk::DeviceMemory) {
        println!("Loading: {:?}", path);

        if !path.exists() {
            panic!("Missing file: {:?}", path);
        }

        if let Ok(mut image) = image::open(path) {
            let (width, height) = image.dimensions();
            let color_type = image.color();

            println!("Width: {}, Height: {}", width, height);

            // TODO: Support multiple image channels, instead of hardcoding it to 4.
            let size = (std::mem::size_of::<u8>() as u32 * width * height * 4) as vk::DeviceSize;

            // Convert texture format if necessary.
            let mut image = match &image {
                image::DynamicImage::ImageLuma8(_) | image::DynamicImage::ImageRgb8(_) => {
                    image.to_rgba8().into()
                }
                image::DynamicImage::ImageLumaA8(_) | image::DynamicImage::ImageRgba8(_) => image,

                _ => todo!("Unsupported image format!"),
            };

            let data = image.as_bytes();

            // Copy image to staging buffer.
            let staging_buffer_create_info = vk::BufferCreateInfo {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let staging_buffer = self.create_buffer(
                staging_buffer_create_info,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            unsafe {
                let data_ptr = self
                    .device
                    .as_ref()
                    .unwrap()
                    .map_memory(
                        staging_buffer.memory,
                        0,
                        staging_buffer_create_info.size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map memory!") as *mut u8;

                data_ptr.copy_from_nonoverlapping(data.as_ptr(), size as usize);

                self.device
                    .as_ref()
                    .unwrap()
                    .unmap_memory(staging_buffer.memory);
            }

            let image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                format: vk::Format::R8G8B8A8_SRGB,
                tiling: vk::ImageTiling::OPTIMAL,
                initial_layout: vk::ImageLayout::UNDEFINED,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Create image
            let image = unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_image(&image_create_info, None)
                    .expect("Failed to create image!")
            };

            // Allocate memory for image.
            // TODO: Abstract this to a `create_image_buffer` helper function instead.

            let texture_memory = unsafe {
                let device = self.device.as_ref().unwrap();

                let memory_requirements = device.get_image_memory_requirements(image);

                let alloc_info = vk::MemoryAllocateInfo {
                    allocation_size: memory_requirements.size,
                    memory_type_index: self.find_memory_type(
                        memory_requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL, // TODO: Could change depending on image use.
                    ),
                    ..Default::default()
                };

                let texture_memory = device
                    .allocate_memory(&alloc_info, None)
                    .expect("Failed to allocate image memory!");

                device.bind_image_memory(image, texture_memory, 0);

                texture_memory
            };

            self.transition_image_layout(
                image,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.copy_buffer_to_image(&staging_buffer, image, width, height);

            self.destroy_buffer(&staging_buffer);

            (image, texture_memory)
        } else {
            panic!("Failed to open file: {:?}", path);
        }
    }

    fn create_texture_image_view(&mut self, image: vk::Image) -> vk::ImageView {
        self.create_image_view(image, vk::Format::R8G8B8A8_SRGB)
    }

    fn create_texture_sampler(&self) -> vk::Sampler {
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0, // TODO: Use hardware limits instead of hardcoded value.
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            mip_lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: 0.0,
            ..Default::default()
        };

        let sampler = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .expect("Failed to create texture sampler!")
        };
        sampler
    }

    fn create_vertex_buffer(&mut self) {
        let staging_buffer_create_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let staging_buffer = self.create_buffer(
            staging_buffer_create_info,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = self
                .device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer.memory,
                    0,
                    staging_buffer_create_info.size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory!") as *mut Vertex;

            data_ptr.copy_from_nonoverlapping(VERTICES.as_ptr(), VERTICES.len());

            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer.memory);
        }

        let vertex_buffer_create_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        self.vertex_buffer = Some(self.create_buffer(
            vertex_buffer_create_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ));

        let size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64;
        self.copy_buffer(&staging_buffer, self.vertex_buffer.as_ref().unwrap(), size)
            .expect("Error while copying buffer");

        self.destroy_buffer(&staging_buffer);
    }

    fn create_index_buffer(&mut self) {
        let staging_buffer_create_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<u16>() * INDICES.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let staging_buffer = self.create_buffer(
            staging_buffer_create_info,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = self
                .device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer.memory,
                    0,
                    staging_buffer_create_info.size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory!") as *mut u16;

            data_ptr.copy_from_nonoverlapping(INDICES.as_ptr(), INDICES.len());

            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer.memory);
        }

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<u16>() * INDICES.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        self.index_buffer = Some(self.create_buffer(
            index_buffer_create_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ));

        let size = (std::mem::size_of::<u16>() * INDICES.len()) as u64;
        self.copy_buffer(&staging_buffer, self.index_buffer.as_ref().unwrap(), size)
            .expect("Error while copying buffer");

        self.destroy_buffer(&staging_buffer);
    }

    fn create_uniform_buffers(&mut self) {
        let buffer_size = size_of::<UniformBufferObject>();

        let mut uniform_buffers = Vec::new();
        let mut uniform_buffers_mapped = Vec::new();

        uniform_buffers.reserve(MAX_FRAMES_IN_FLIGHT);
        uniform_buffers_mapped.reserve(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let create_info = vk::BufferCreateInfo {
                size: buffer_size as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = self.create_buffer(
                create_info,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            // TODO: Throws a `Attempting to map memory on an already-mapped` error! Seems unused
            // unsafe {
            //     self.device
            //         .as_ref()
            //         .unwrap()
            //         .map_memory(
            //             buffer.memory,
            //             0,
            //             buffer_size as u64,
            //             vk::MemoryMapFlags::empty(),
            //         )
            //         .expect("Failed to map memory!")
            // };

            uniform_buffers_mapped.push(buffer.memory);
            uniform_buffers.push(buffer);
        }

        self.uniform_buffers = Some(uniform_buffers);
        self.uniform_buffers_mapped = Some(uniform_buffers_mapped);
    }

    fn create_descriptor_pool(&mut self) {
        let pool_size = [vk::DescriptorPoolSize {
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        }];

        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: 1,
            p_pool_sizes: pool_size.as_ptr(),
            max_sets: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        self.descriptor_pool = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_pool(&pool_info, None)
                    .expect("Failed to create descriptor pool!"),
            )
        };
    }

    fn create_descriptor_sets(&mut self) {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..self.swapchain_images.as_ref().unwrap().len() {
            layouts.push(*self.descriptor_set_layout.as_ref().unwrap());
        }

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: *self.descriptor_pool.as_ref().unwrap(),
            descriptor_set_count: MAX_FRAMES_IN_FLIGHT as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to create descriptor sets!")
        };

        self.descriptor_sets = Some(descriptor_sets);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: self.uniform_buffers.as_ref().unwrap()[0].buffer,
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as u64,
            };

            let descriptor_write = vk::WriteDescriptorSet {
                dst_set: self.descriptor_sets.as_ref().unwrap()[i],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                p_buffer_info: &buffer_info,
                ..Default::default()
            };

            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&[descriptor_write], &[])
            }
        }
    }

    fn create_command_pool(&mut self, indices: &QueueFamiliyIndices) {
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: indices.graphics_family.unwrap(),
            ..Default::default()
        };

        self.command_pool = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_command_pool(&create_info, None)
                    .expect("Failed to create command pool!"),
            )
        }
    }

    fn create_command_buffers(&mut self) {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: *self.command_pool.as_ref().unwrap(),
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        self.command_buffers = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to allocate command buffers"),
            )
        }
    }

    fn create_sync_objects(&mut self) {
        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let device = self.device.as_ref().unwrap();

        (
            self.image_available_semaphores,
            self.render_finished_semaphores,
            self.in_flight_fences,
        ) = unsafe {
            let mut image_available_semaphores = Vec::new();
            let mut render_finished_semaphores = Vec::new();
            let mut in_flight_fences = Vec::new();

            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                image_available_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore!"),
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore!"),
                );
                in_flight_fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .expect("Failed to create fence!"),
                );
            }

            (
                Some(image_available_semaphores),
                Some(render_finished_semaphores),
                Some(in_flight_fences),
            )
        };
    }

    fn record_command_buffer(&mut self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo {
            ..Default::default()
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin recording command buffer!");
        }

        let clear_color = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: *self.render_pass.as_ref().unwrap(),
            framebuffer: self.swapchain_frame_buffers.as_ref().unwrap()[image_index],
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: *self.swapchain_extent.as_ref().unwrap(),
            },
            clear_value_count: 1,
            p_clear_values: clear_color.as_ptr(),
            ..Default::default()
        };

        unsafe {
            let device = self.device.as_ref().unwrap();

            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *self.graphics_pipeline.as_ref().unwrap(),
            );

            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain_extent.as_ref().unwrap().width as f32,
                height: self.swapchain_extent.as_ref().unwrap().height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            device.cmd_set_viewport(command_buffer, 0, &viewport);

            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: *self.swapchain_extent.as_ref().unwrap(),
            }];
            device.cmd_set_scissor(command_buffer, 0, &scissor);

            let vertex_buffers = [self.vertex_buffer.as_ref().unwrap().buffer];
            let offsets = [0];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.as_ref().unwrap().buffer,
                0,
                vk::IndexType::UINT16,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline_layout.as_ref().unwrap(),
                0,
                &[self.descriptor_sets.as_ref().unwrap()[self.frame_index]],
                &[],
            );
            device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

            device.cmd_end_render_pass(command_buffer);
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer!");
        }
    }

    fn cleanup_swap_chain(&mut self) {
        let device = self.device.as_ref().unwrap();

        unsafe {
            for frame_buffer in self.swapchain_frame_buffers.as_ref().unwrap() {
                device.destroy_framebuffer(*frame_buffer, None);
            }
            for image_view in self.swapchain_image_views.as_ref().unwrap() {
                device.destroy_image_view(*image_view, None);
            }
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(*self.swapchain.as_ref().unwrap(), None);
        }
    }

    fn recreate_swap_chain(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("Error while waiting for device idle!");
        }

        self.cleanup_swap_chain();

        let physical_device = *self.physical_device.as_ref().unwrap();
        self.create_swap_chain(&physical_device);
        self.create_swapchain_image_views();
        self.create_frame_buffers();
    }

    fn init_vulkan(&mut self, window: &Window) {
        self.entry = Some(Entry::linked());
        self.instance = Some(Self::create_instance(self.entry.as_ref().unwrap(), window));
        self.init_debug_messenger();
        let (surface, surface_loader) = self.create_surface(window);
        self.surface = Some(surface);
        self.surface_loader = Some(surface_loader);

        self.physical_device = Some(
            self.get_physical_device()
                .expect("Error while getting physical device"),
        );
        let physical_device = *self.physical_device.as_ref().unwrap();

        let indices = self.create_logical_device(&physical_device);
        self.create_swap_chain(&physical_device);
        self.create_swapchain_image_views();
        self.create_render_pass();
        self.create_descriptor_set_layout();
        let (pipeline, layout) = self.create_graphics_pipeline();
        self.graphics_pipeline = Some(pipeline);
        self.pipeline_layout = Some(layout);
        self.create_frame_buffers();
        self.create_command_pool(&indices);
        self.image_textures = Some(vec![self.create_texture_image(Path::new(TEXTURE_PATH))]);
        self.texture_image_view =
            Some(self.create_texture_image_view(self.image_textures.as_ref().unwrap()[0].0));
        self.texture_sampler = Some(self.create_texture_sampler());
        self.create_vertex_buffer();
        self.create_index_buffer();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_descriptor_sets();
        self.create_command_buffers();
        self.create_sync_objects();
    }

    fn render(&mut self) {
        // Do cool stuff here.
        unsafe {
            let fences = [self.in_flight_fences.as_ref().unwrap()[self.frame_index]];

            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait for Fence!");

            let (image_index, is_sub_optimal) = self
                .swapchain_loader
                .as_ref()
                .unwrap()
                .acquire_next_image(
                    *self.swapchain.as_ref().unwrap(),
                    u64::MAX,
                    self.image_available_semaphores.as_ref().unwrap()[self.frame_index],
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image.");

            if is_sub_optimal {
                return self.recreate_swap_chain();
            }

            let device = self.device.as_ref().unwrap();

            #[cfg(feature = "verbose-frame-time")]
            {
                println!("Rendering: {} ms", self.delta_time * 1000.0);
            }

            let wait_semaphores =
                [self.image_available_semaphores.as_ref().unwrap()[self.frame_index]];
            let signal_semaphores =
                [self.render_finished_semaphores.as_ref().unwrap()[self.frame_index]];

            device
                .reset_fences(&fences)
                .expect("Failed to reset fence!");

            device
                .reset_command_buffer(
                    self.command_buffers.as_ref().unwrap()[self.frame_index],
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer!");

            self.update_uniform_buffer(self.frame_index);

            self.record_command_buffer(
                self.command_buffers.as_ref().unwrap()[self.frame_index],
                image_index as usize,
            );

            let submit_info = [vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT].as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: [self.command_buffers.as_ref().unwrap()[self.frame_index]]
                    .as_ptr(),
                signal_semaphore_count: 1,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            }];

            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    *self.graphics_queue.as_ref().unwrap(),
                    &submit_info,
                    self.in_flight_fences.as_ref().unwrap()[self.frame_index],
                )
                .expect("Failed to submit draw command buffer!");

            let swapchains = [*self.swapchain.as_ref().unwrap()];
            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: &image_index,
                ..Default::default()
            };

            let result = self
                .swapchain_loader
                .as_ref()
                .unwrap()
                .queue_present(*self.present_queue.as_ref().unwrap(), &present_info);

            match result {
                Ok(_) => {}
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        self.recreate_swap_chain()
                    }
                    _ => panic!("Failed to execute queue present."),
                },
            }

            self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    pub fn update_uniform_buffer(&mut self, frame_index: usize) {
        self.time += self.delta_time;

        let ubo = [UniformBufferObject {
            model: glam::Mat4::from_rotation_z(self.time * 90.0_f32.to_radians()),
            view: glam::Mat4::look_at_rh(
                Vec3 {
                    x: 2.0,
                    y: 2.0,
                    z: 2.0,
                },
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ),
            proj: glam::Mat4::perspective_rh(
                45.0_f32.to_radians(),
                self.swapchain_extent.as_ref().unwrap().width as f32
                    / self.swapchain_extent.as_ref().unwrap().height as f32,
                0.1,
                100.0,
            ),
        }];

        let size = std::mem::size_of::<UniformBufferObject>();

        unsafe {
            let data_ptr =
                self.device
                    .as_ref()
                    .unwrap()
                    .map_memory(
                        self.uniform_buffers.as_ref().unwrap()[frame_index].memory,
                        0,
                        size as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map memory!") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubo.as_ptr(), ubo.len());

            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(self.uniform_buffers.as_ref().unwrap()[self.frame_index].memory);
        }
    }

    pub fn shutdown(&mut self) {
        println!("Shutdown called");
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("Error while waiting for device idle!");

            self.cleanup_swap_chain();

            let device = self.device.as_ref().unwrap();

            // TODO: Replace with `if let Some()`
            device.destroy_sampler(self.texture_sampler.unwrap(), None);
            device.destroy_image_view(self.texture_image_view.unwrap(), None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.destroy_buffer(&self.uniform_buffers.as_ref().unwrap()[i]);
            }

            for texture in self.image_textures.as_ref().unwrap() {
                device.destroy_image(texture.0, None);
                device.free_memory(texture.1, None);
            }

            device.destroy_descriptor_pool(*self.descriptor_pool.as_ref().unwrap(), None);

            device
                .destroy_descriptor_set_layout(*self.descriptor_set_layout.as_ref().unwrap(), None);

            self.destroy_buffer(self.index_buffer.as_ref().unwrap());
            self.destroy_buffer(self.vertex_buffer.as_ref().unwrap());

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                device
                    .destroy_semaphore(self.image_available_semaphores.as_ref().unwrap()[i], None);
                device
                    .destroy_semaphore(self.render_finished_semaphores.as_ref().unwrap()[i], None);
                device.destroy_fence(self.in_flight_fences.as_ref().unwrap()[i], None);
            }

            device.destroy_command_pool(*self.command_pool.as_ref().unwrap(), None);

            device.destroy_pipeline(*self.graphics_pipeline.as_ref().unwrap(), None);
            device.destroy_pipeline_layout(*self.pipeline_layout.as_ref().unwrap(), None);
            device.destroy_render_pass(*self.render_pass.as_ref().unwrap(), None);

            device.destroy_device(None);
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
