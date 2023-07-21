use cgmath::{Matrix4, SquareMatrix, Vector3, Vector4};
use image::{DynamicImage, ImageBuffer};
use vulkano::{
    buffer::{
        self,
        BufferContents,
        BufferCreateInfo,
        BufferUsage,
        Subbuffer,
    },
    image::ImageDimensions,
    memory::allocator::{
        AllocationCreateInfo,
        MemoryAllocator,
        MemoryUsage,
    },
};

pub struct Texture {
    pub data: Vec<u8>,
    pub dimensions: ImageDimensions,
}

type TextureID = usize;

pub struct Material {
    pub name: Option<String>,

    pub base_color_factor: Vector4<f32>,
    pub base_color_texture: Option<TextureID>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<TextureID>,

    pub normal_texture: Option<TextureID>,
    pub normal_scale: Option<f32>,

    pub occlusion_texture: Option<TextureID>,
    pub occlusion_strength: Option<f32>,

    pub emissive_factor: Vector3<f32>,
    pub emissive_texture: Option<TextureID>,

    pub alpha_cutoff: Option<f32>,
    pub alpha_mode: gltf::material::AlphaMode,

    pub double_sided: bool,
}

pub struct Buffer {
    pub positions: Subbuffer<[f32]>,
    pub normals: Subbuffer<[f32]>,
    pub texcoords: Subbuffer<[f32]>,
    pub indices: Subbuffer<[u32]>,
    pub meshes: Vec<Vec<Primitive>>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
}

pub struct Primitive {
    pub index_offset: usize,
    pub index_count: usize,
    pub vert_offset: usize,
    pub mat_idx: usize,
}

pub struct BufferBuilder {
    positions: Vec<f32>,
    normals: Vec<f32>,
    texcoords: Vec<f32>,
    indices: Vec<u32>,
    meshes: Vec<Vec<Primitive>>,
    textures: Vec<Texture>,
    materials: Vec<Material>,
}

pub struct Object {
    pub transform: [[f32; 4]; 4],
    pub mesh_id: usize,
    pub name: String,
}

impl BufferBuilder {
    pub fn new() -> Self {
        BufferBuilder {
            positions: Vec::new(),
            normals: Vec::new(),
            texcoords: Vec::new(),
            indices: Vec::new(),
            meshes: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn load_gltf(
        mut self,
        path: &str,
        objects: &mut Vec<Object>,
    ) -> BufferBuilder {
        let (gltf, buffers, images) = gltf::import(path).expect("Failed to read GLTF file");

        let tex_offset = self.textures.len();
        let mat_offset = self.materials.len();
        let mesh_offset = self.meshes.len();

        self.textures.extend(
            images.into_iter()
                .map(|image| {
                    let width = image.width;
                    let height = image.height;
                    let pixels = image.pixels;

                    let data = match image.format {
                        gltf::image::Format::R8 => DynamicImage::ImageLuma8(
                            ImageBuffer::from_raw(width, height, pixels)
                                .expect("Failed to load image data")),
                        gltf::image::Format::R8G8 => DynamicImage::ImageLumaA8(
                            ImageBuffer::from_raw(width, height, pixels)
                                .expect("Failed to load image data")),
                        gltf::image::Format::R8G8B8 => DynamicImage::ImageRgb8(
                            ImageBuffer::from_raw(width, height, pixels)
                                .expect("Failed to load image data")),
                        gltf::image::Format::R8G8B8A8 => DynamicImage::ImageRgba8(
                            ImageBuffer::from_raw(width, height, pixels)
                                .expect("Failed to load image data")),
                        _ => panic!("Unsupported image format: {:?}", image.format)
                    }.to_rgba8().as_raw().to_owned();

                    let dimensions = ImageDimensions::Dim2d {
                        width,
                        height,
                        array_layers: 1,
                    };

                    Texture {
                        data,
                        dimensions,
                    }
                })
        );

        self.materials.extend(
            gltf.materials()
                .map(|mat| {
                    let pbr = mat.pbr_metallic_roughness();

                    Material {
                        name: mat.name().map(Into::into),

                        base_color_factor: pbr.base_color_factor().into(),
                        base_color_texture: pbr.base_color_texture().map(|tex| tex_offset + tex.texture().source().index()),
                        metallic_factor: pbr.metallic_factor(),
                        roughness_factor: pbr.roughness_factor(),
                        metallic_roughness_texture: pbr.metallic_roughness_texture().map(|tex| tex_offset + tex.texture().source().index()),

                        normal_texture: mat.normal_texture().map(|tex| tex_offset + tex.texture().source().index()),
                        normal_scale: mat.normal_texture().map(|tex| tex.scale()),

                        occlusion_texture: mat.occlusion_texture().map(|tex| tex_offset + tex.texture().source().index()),
                        occlusion_strength: mat.occlusion_texture().map(|tex| tex.strength()),

                        emissive_factor: mat.emissive_factor().into(),
                        emissive_texture: mat.emissive_texture().map(|tex| tex_offset + tex.texture().source().index()),

                        alpha_cutoff: mat.alpha_cutoff(),
                        alpha_mode: mat.alpha_mode(),

                        double_sided: mat.double_sided(),
                    }
                })
        );

        self.meshes.extend(
            gltf.meshes()
                .map(|mesh| mesh.primitives())
                .map(|prims| {
                    prims
                        .map(|prim| {
                            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                            let vert_offset = self.positions.len() / 3;
                            self.positions.extend(
                                reader.read_positions()
                                    .expect("Unable to read positions from mesh")
                                    .flatten()
                            );
                            self.normals.extend(
                                reader.read_normals()
                                    .expect("Unable to read normals from mesh")
                                    .flatten()
                            );
                            self.texcoords.extend(
                                reader.read_tex_coords(0)
                                    .expect("Unable to read texcoords from mesh")
                                    .into_f32()
                                    .flatten()
                            );
                            let index_offset = self.indices.len();
                            self.indices.extend(
                                reader.read_indices()
                                    .expect("Unable to read indices from mesh")
                                    .into_u32()
                            );
                            let index_count = self.indices.len() - index_offset;

                            Primitive {
                                index_offset,
                                index_count,
                                vert_offset,
                                mat_idx: prim.material().index().map(|i| i + mat_offset).unwrap(), // TODO: support models without materials
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<Vec<_>>>()
        );

        let nodes = gltf.nodes()
            .enumerate()
            .map(|(index, node)| {
                assert_eq!(node.index(), index);
                (
                    node.name().unwrap_or("").to_string(),
                    Matrix4::from(node.transform().matrix()),
                    node.mesh().map(|m| mesh_offset + m.index()),
                    node.children()
                        .map(|child| child.index())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        println!("Loaded {} nodes", nodes.len());

        {
            let mut stack = gltf.scenes()
                .flat_map(|scene| {
                    scene.nodes()
                        .map(|node| {
                            (node.index(), Matrix4::identity())
                        })
                })
                .collect::<Vec<_>>();

            while let Some((node_id, parent_transform)) = stack.pop() {
                let (name, node_transform, mesh_id, children) = nodes[node_id].clone();

                let transform = parent_transform * node_transform;

                if let Some(mesh_id) = mesh_id {
                    objects.push(Object {
                        transform: transform.into(),
                        mesh_id,
                        name,
                    });
                }

                for child_id in children {
                    stack.push((child_id, transform));
                }
            }
        };

        self
    }

    pub fn build(self, memory_allocator: &impl MemoryAllocator) -> Buffer {
        Buffer {
            positions: Self::make_buffer(self.positions.into_iter(), BufferUsage::VERTEX_BUFFER, memory_allocator),
            normals: Self::make_buffer(self.normals.into_iter(), BufferUsage::VERTEX_BUFFER, memory_allocator),
            texcoords: Self::make_buffer(self.texcoords.into_iter(), BufferUsage::VERTEX_BUFFER, memory_allocator),
            indices: Self::make_buffer(self.indices.into_iter(), BufferUsage::INDEX_BUFFER, memory_allocator),
            meshes: self.meshes,
            textures: self.textures,
            materials: self.materials,
        }
    }

    fn make_buffer<T, I>(
        data: I,
        usage: BufferUsage,
        memory_allocator: &impl MemoryAllocator,
    ) -> Subbuffer<[T]>
        where
            T: BufferContents,
            I: IntoIterator<Item=T>,
            I::IntoIter: ExactSizeIterator,
    {
        buffer::Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            data,
        ).unwrap()
    }
}
