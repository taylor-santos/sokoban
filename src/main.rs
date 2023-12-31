// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{sync::Arc, time::Instant};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::time::Duration;

use cgmath::{Deg, Euler, Matrix4, One, Point3, Rad, Vector3};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, DeviceExtensions, physical::PhysicalDeviceType, QueueCreateInfo, QueueFlags},
    format::Format,
    image::{AttachmentImage, ImageAccess, ImageUsage, ImmutableImage, MipmapsCount, SwapchainImage, view::ImageView},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    swapchain::{
        acquire_next_image, AcquireError, PresentMode, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowBuilder},
};

use camera::FirstPersonCamera;
use crate::mesh::Object;

mod mesh;
mod texture;
mod camera;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Texcoord {
    #[format(R32G32_SFLOAT)]
    texcoord: [f32; 2],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct InstanceData {
    #[format(R32G32B32A32_SFLOAT)]
    pub transform_instance: [[f32; 4]; 4],
}

// An object bundled with a descriptor set for its transformation
type ObjectSet = (Object, Arc<PersistentDescriptorSet>);

struct InstanceObjects {
    objects: Vec<ObjectSet>,
    offset: u32,
    count: u32,
}

fn main() {
    // The first step of any Vulkan program is to create an instance.
    let instance = {
        let library = VulkanLibrary::new().unwrap();
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
        // required to draw to a window.
        let required_extensions = vulkano_win::required_extensions(&library);
        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                // Enable enumerating devices that use non-conformant Vulkan implementations. (e.g.
                // MoltenVK)
                enumerate_portability: true,
                ..Default::default()
            },
        ).unwrap()
    };

    // Next, we need to create the window.
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform
    // winit window and a cross-platform Vulkan surface that represents the surface of the window.
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Sokoban Game")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let window = surface.object()
        .unwrap()
        .downcast_ref::<Window>()
        .unwrap();

    // Choose device extensions that we're going to use. In order to present images to a surface,
    // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // We then choose which physical device to use. First, we enumerate all the available physical
    // devices, then apply filters to narrow them down to those that can support our needs.
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // Some devices may not support the extensions or features that your application, or
            // report properties and limits that are not sufficient for your application. These
            // should be filtered out here.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will execute
            // our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a draw
            // queue and a compute queue), similar to CPU threads. This is something you have to
            // have to manage manually in Vulkan. Queues of the same type belong to the same queue
            // family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-world application, you may want to use a separate dedicated transfer queue to
            // handle data transfers in parallel with graphics operations. You may also need a
            // separate queue for compute operations, if your application uses those.
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // We select a queue family that supports graphics operations. When drawing to
                    // a window surface, as we do in this example, we also need to check that
                    // queues in this queue family are capable of presenting images to the surface.
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|i| (p, i as u32))
        })
        // All the physical devices that pass the filters above are suitable for the application.
        // However, not every device is equal, some are preferred over others. Now, we assign each
        // physical device a score, and pick the device with the lowest ("best") score.
        //
        // In this example, we simply select the best-scoring device to use in the application.
        // In a real-world setting, you may want to use the best-scoring device only as a "default"
        // or "recommended" device, and let the user choose the device themself.
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // An iterator of created queues is returned by the function alongside the device.
    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            // A list of optional features and extensions that our program needs to work correctly.
            // Some parts of the Vulkan specs are optional and must be enabled manually at device
            // creation. In this example the only thing we are going to need is the `khr_swapchain`
            // extension that allows us to draw to a window.
            enabled_extensions: device_extensions,

            // The list of queues that we are going to use. Here we only use one queue, from the
            // previously chosen queue family.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            ..Default::default()
        },
    )
        .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. We only
    // use one queue in this example, so we just retrieve the first and only element of the
    // iterator.
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
    // swapchain allocates the color buffers that will contain the image that will ultimately be
    // visible on the screen. These images are returned alongside the swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only pass
        // values that are allowed by the capabilities.
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,

                image_format,

                // The dimensions of the window, only used to initially setup the swapchain.
                //
                // NOTE:
                // On some drivers the swapchain dimensions are specified by
                // `surface_capabilities.current_extent` and the swapchain size must use these
                // dimensions. These dimensions are always the same as the window dimensions.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window dimensions.
                //
                // Both of these cases need the swapchain to use the window dimensions, so we just
                // use that.
                image_extent: window.inner_size().into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,

                // The present mode determines how the swapchain behaves when multiple images are
                // waiting in the queue to be presented.
                //
                // `PresentMode::Immediate` (vsync off) displays the latest image immediately,
                // without waiting for the next vertical blanking period. This may cause tearing.
                //
                // `PresentMode::Fifo` (vsync on) appends the latest image to the end of the queue,
                // and the front of the queue is removed during each vertical blanking period to be
                // presented. No tearing will be visible.
                present_mode: PresentMode::Immediate,

                // The alpha mode indicates how the alpha value of the final image will behave. For
                // example, you can choose whether the window will be opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
            .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let subbuffer_allocator = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
    );

    // Load the vertex and fragment shaders, respectively
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images where the
    // colors, depth and/or stencil information will be written.
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this attachment
                // at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw in the
                // actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to be one
                // of the types of the `vulkano::format` module (or alternatively one of your
                // structs that implements the `FormatDesc` trait). Here we use the same format as
                // the swapchain.
                format: swapchain.image_format(),
                // `samples: 1` means that we ask the GPU to use one sample to determine the value
                // of each pixel in the color attachment. We could use a larger value
                // (multisampling) for antialiasing. An example of this can be found in
                // msaa-renderpass.rs.
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D32_SFLOAT,
                samples: 1,
            },
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            depth_stencil: {depth},
        },
    )
        .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state([
            Position::per_vertex(),
            Normal::per_vertex(),
            Texcoord::per_vertex(),
            InstanceData::per_instance(),
        ])
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        // A rasterization_state is necessary to render textures
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(
        &memory_allocator,
        &images,
        render_pass.clone(),
        &mut viewport,
    );

    let (instances, instance_buffer, meshes) = {
        let mut builder = mesh::BufferBuilder::new();

        let mut instances = Vec::<InstanceObjects>::new();
        let mut instance_data = Vec::<InstanceData>::new();

        // Helper function to store a set of objects' instances in the above vectors
        let mut add_instances = |is: Vec<InstanceData>, objects: Vec<ObjectSet>| {
            instances.push(InstanceObjects {
                objects,
                offset: instance_data.len() as u32,
                count: is.len() as u32,
            });
            let mut is = is;
            instance_data.append(&mut is);
        };

        // Helper function to build a descriptor set for an object's transform
        let create_object_descriptor_set = |object: Object| -> ObjectSet {
            let transform = object.transform;
            let object_subbuffer = {
                let subbuffer = subbuffer_allocator.allocate_sized().unwrap();
                let data = vs::Object {
                    transform,
                };
                *subbuffer.write().unwrap() = data;
                subbuffer
            };
            let transform_layout = pipeline.layout().set_layouts().get(2).unwrap();
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                transform_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, object_subbuffer),
                ],
            ).expect("Failed to create descriptor set");

            (object, set)
        };

        {
            let mut sponza_objects = Vec::<Object>::new();
            builder = builder.load_gltf("models/sponza.glb", &mut sponza_objects);
            let sponza_objects = sponza_objects.into_iter()
                .map(create_object_descriptor_set)
                .collect::<Vec<_>>();

            // For this model, create one instance with just the identity matrix applied
            let sponza_instances = vec!(InstanceData {
                transform_instance: Matrix4::one().into(),
            });

            add_instances(sponza_instances, sponza_objects);
        }
        {
            let mut helmet_objects = Vec::<Object>::new();
            builder = builder.load_gltf("models/DamagedHelmet.glb", &mut helmet_objects);
            let helmet_objects = helmet_objects.into_iter()
                .map(create_object_descriptor_set)
                .collect::<Vec<_>>();

            // For this model, create a 10x10x10 grid of instances with a different rotation applied to each
            let mut helmet_instances = Vec::<InstanceData>::new();
            for (x,y,z) in itertools::iproduct!(-5..5, 0..10, -5..5) {
                helmet_instances.push(InstanceData {
                    transform_instance: (
                        Matrix4::from_translation(Vector3::new(x as f32, y as f32, z as f32)) *
                            Matrix4::from_scale(0.1) *
                            Matrix4::from(Euler::new(
                                Deg(x as f32) * 100.0,
                                Deg(y as f32) * 100.0,
                                Deg(z as f32) * 100.0,
                            ))
                    ).into(),
                });
            }

            add_instances(helmet_instances, helmet_objects);
        }

        let meshes = builder.build(&memory_allocator);

        let instance_buffer = {
            Buffer::from_iter(
                &memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                instance_data,
            )
                .unwrap()
        };

        (instances, instance_buffer, meshes)
    };

    let texture_sets = {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();
        let image_buffer_time = Instant::now();
        let images = meshes.textures.into_iter().map(|texture| {
            let image = ImmutableImage::from_iter(
                &memory_allocator,
                texture.data.into_iter(),
                texture.dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_UNORM,
                &mut builder,
            ).unwrap();
            ImageView::new_default(image)
                .unwrap()
        })
            .collect::<Vec<_>>();

        dbg!(image_buffer_time.elapsed());

        builder.build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
            .unwrap();

        let texture_layout = pipeline.layout().set_layouts().get(1).unwrap();

        images.iter()
            .map(|image| {
                PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    texture_layout.clone(),
                    [
                        WriteDescriptorSet::image_view_sampler(0, image.clone(), sampler.clone()),
                    ],
                ).expect("Failed to create descriptor set")
            })
            .collect::<Vec<_>>()
    };

    let mut camera = FirstPersonCamera::new();
    camera.position = Point3::new(6.0, 1.5, -0.5);
    camera.yaw = Rad(FRAC_PI_2);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let mut last_instant = Instant::now();
    let mut forward = false;
    let mut right = false;
    let mut backward = false;
    let mut left = false;

    let mut frame_times = Vec::<Duration>::new();
    let frame_report_frequency = 1000;

    set_cursor_confinement(window, false);
    let mut mouse_attached = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(_) => recreate_swapchain = true,
            WindowEvent::MouseInput { button: MouseButton::Left, .. } => {
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                set_cursor_confinement(window, true);
                mouse_attached = true;
            }
            WindowEvent::KeyboardInput { input, .. } => {
                let pressed = input.state == ElementState::Pressed;
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

                match input.virtual_keycode {
                    Some(VirtualKeyCode::W) => forward = pressed,
                    Some(VirtualKeyCode::A) => left = pressed,
                    Some(VirtualKeyCode::S) => backward = pressed,
                    Some(VirtualKeyCode::D) => right = pressed,
                    Some(VirtualKeyCode::Q) => *control_flow = ControlFlow::Exit,
                    Some(VirtualKeyCode::Escape) => {
                        set_cursor_confinement(window, false);
                        mouse_attached = false;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
            if mouse_attached {
                camera.rotate(delta.0 as f32, delta.1 as f32);
            }
        }
        Event::MainEventsCleared => {
            let now = Instant::now();
            let delta_time = now.duration_since(last_instant);
            last_instant = now;

            frame_times.push(delta_time);
            if frame_times.len() >= frame_report_frequency {
                let total_time: Duration = frame_times.iter().sum();
                let frame_time = total_time / frame_times.len() as u32;
                let seconds = frame_time.as_secs() as f64;
                let nanos = frame_time.subsec_nanos() as f64;
                let ms = seconds * 1000.0 + nanos / 1_000_000.0;
                let fps = frame_times.len() as f64 / total_time.as_secs_f64();
                println!("Last {} frames avg: {}ms ({} fps)", frame_times.len(), ms, fps);
                frame_times.clear();
            }


            let delta_t = delta_time.as_secs_f32();

            if forward { camera.move_forward(delta_t); }
            if left { camera.move_left(delta_t); }
            if right { camera.move_right(delta_t); }
            if backward { camera.move_backward(delta_t); }
        }
        Event::RedrawEventsCleared => {
            // Do not draw the frame when the screen dimensions are zero. On Windows, this can
            // occur when minimizing the application.
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let dimensions = window.inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            // It is important to call this function from time to time, otherwise resources
            // will keep accumulating and you will eventually reach an out of memory error.
            // Calling this function polls various fences in order to determine what the GPU
            // has already processed, and frees the resources that are no longer needed.
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            // Whenever the window resizes we need to recreate everything dependent on the
            // window size. In this example that includes the swapchain, the framebuffers and
            // the dynamic state viewport.
            if recreate_swapchain {
                // Use the new dimensions of the window.
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    // This error tends to happen when the user is manually resizing the
                    // window. Simply restarting the loop is the easiest way to fix this
                    // issue.
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("failed to recreate swapchain: {e}"),
                };

                swapchain = new_swapchain;
                framebuffers = window_size_dependent_setup(
                    &memory_allocator,
                    &new_images,
                    render_pass.clone(),
                    &mut viewport,
                );
                recreate_swapchain = false;
            }

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            // `acquire_next_image` can be successful, but suboptimal. This means that the
            // swapchain image will still work, but it may not display correctly. With some
            // drivers this can be when the window resizes, but it may not cause the swapchain
            // to become out of date.
            if suboptimal {
                recreate_swapchain = true;
            }

            // In order to draw, we have to build a *command buffer*. The command buffer object
            // holds the list of commands that are going to be executed.
            //
            // Building a command buffer is an expensive operation (usually a few hundred
            // microseconds), but it is known to be a hot path in the driver and is expected to
            // be optimized.
            //
            // Note that we have to pass a queue family when we create the command buffer. The
            // command buffer will only be executable on that given queue family.
            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
                .unwrap();

            builder
                .bind_vertex_buffers(0, (
                    meshes.positions.clone(),
                    meshes.normals.clone(),
                    meshes.texcoords.clone(),
                    instance_buffer.clone(),
                ))
                .bind_index_buffer(meshes.indices.clone());

            let clear_values = vec![Some([0.0, 0.68, 1.0, 1.0].into()), Some(1.0.into())];

            let uniform_subbuffer = {
                let proj = {
                    let aspect_ratio = swapchain.image_extent()[0] as f32
                        / swapchain.image_extent()[1] as f32;
                    let near = 0.005;
                    let far = 10000.0;

                    let proj = cgmath::perspective(
                        Rad(FRAC_PI_4),
                        aspect_ratio,
                        near,
                        far,
                    );
                    // Vulkan clip space has inverted Y and half Z, compared with OpenGL.
                    // A corrective transformation is needed to make an OpenGL perspective matrix
                    // work properly. See here for more info:
                    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
                    let correction = Matrix4::<f32>::new(
                        1.0, 0.0, 0.0, 0.0,
                        0.0, -1.0, 0.0, 0.0,
                        0.0, 0.0, 0.5, 0.0,
                        0.0, 0.0, 0.5, 1.0,
                    );

                    correction * proj
                };
                let view = camera.get_view_matrix();

                let uniform_data = vs::Data {
                    view: view.into(),
                    proj: proj.into(),
                    camera_pos: camera.position.into(),
                };

                let subbuffer = subbuffer_allocator.allocate_sized().unwrap();
                *subbuffer.write().unwrap() = uniform_data;

                subbuffer
            };

            let layout0 = pipeline.layout().set_layouts().get(0).unwrap();
            let camera_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout0.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_subbuffer),
                ],
            ).unwrap();

            builder
                // Before we can draw, we have to *enter a render pass*.
                .begin_render_pass(
                    RenderPassBeginInfo {
                        // A list of values to clear the attachments with. This list contains
                        // one item for each attachment in the render pass. In this case, there
                        // are two attachments, color and depth, and we clear them with a blue
                        // color and 1.0, respectively.
                        // Only attachments that have `LoadOp::Clear` are provided with clear
                        // values, any others should use `ClearValue::None` as the clear value.
                        clear_values,
                        ..RenderPassBeginInfo::framebuffer(
                            framebuffers[image_index as usize].clone(),
                        )
                    },
                    // The contents of the first (and only) subpass. This can be either
                    // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more advanced
                    // and is not covered here.
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone());

            for instance in &instances {
                for (object, transform_set) in &instance.objects {
                    let primitives = &meshes.meshes[object.mesh_id];
                    for prim in primitives {
                        let mat = &meshes.materials[prim.mat_idx];
                        let tex_idx = mat.base_color_texture.unwrap();
                        let texture_set = texture_sets[tex_idx].clone();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                pipeline.layout().clone(),
                                0,
                                (camera_set.clone(), texture_set.clone(), transform_set.clone()),
                            )
                            .draw_indexed(
                                prim.index_count as u32,
                                instance.count,
                                prim.index_offset as u32,
                                prim.vert_offset as i32,
                                instance.offset,
                            )
                            .unwrap();
                    }
                }
            }

            builder
                // We leave the render pass. Note that if we had multiple subpasses we could
                // have called `next_subpass` to jump to the next subpass.
                .end_render_pass()
                .unwrap();

            // Finish building the command buffer by calling `build`.
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                // The color output is now expected to contain our rendered scene. But in order to
                // show it on the screen, we have to *present* the image by calling
                // `then_swapchain_present`.
                //
                // This function does not actually present the image immediately. Instead it
                // submits a present command at the end of the queue. This means that it will
                // only be presented once the GPU has finished executing the command buffer
                // that draws the scene.
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    allocator: &StandardMemoryAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(allocator, dimensions, Format::D32_SFLOAT).unwrap(),
    )
        .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn set_cursor_confinement(
    window: &Window,
    state: bool,
) {
    if state {
        window.set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
            .unwrap();
    } else {
        window.set_cursor_grab(CursorGrabMode::None).unwrap();
    }
    // No cursor if mouse is confined
    window.set_cursor_visible(!state);
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
    }
}
