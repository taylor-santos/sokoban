use std::io::Cursor;
use std::time::Instant;

use anyhow::{anyhow, Error};
use cgmath::{Matrix4, SquareMatrix, Vector3, Vector4};
use dataurl::DataUrl;
use gltf::image::Source;
use image::RgbaImage;
use vulkano::{
    buffer::{
        Buffer,
        BufferContents,
        BufferCreateInfo,
        BufferUsage,
        Subbuffer,
    },
    memory::allocator::{
        AllocationCreateInfo,
        MemoryAllocator,
        MemoryUsage,
    },
};
use vulkano::image::ImageDimensions;

use texture::load_texture;

use crate::texture;

pub struct Mesh {
    pub vertices: Subbuffer<[f32]>,
    pub normals: Option<Subbuffer<[f32]>>,
    pub texcoords: Option<Subbuffer<[f32]>>,
    pub indices: Subbuffer<[u32]>,
    pub material: Option<usize>,
}

pub struct Texture {
    pub data: RgbaImage,
    pub dimensions: ImageDimensions,
}

type TextureID = usize;

pub struct Material {
    pub index: Option<usize>,
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

#[derive(Debug)]
pub struct Object {
    pub transform: [[f32; 4]; 4],
    pub mesh_id: usize,
    pub name: String,
}

impl Mesh {
    pub fn from_gltf(
        path: &str,
        memory_allocator: &impl MemoryAllocator,
    ) -> Result<(Vec<Vec<Mesh>>, Vec<Material>, Vec<Texture>, Vec<Object>), Error> {
        let loading_gltf_time = Instant::now();
        let (gltf, buffers, _) = gltf::import(path).expect("Failed to read GLTF file");
        println!("Loading GLTF model {}", path);

        let textures = gltf.textures()
            .enumerate()
            .map(|(index, texture)| {
                assert_eq!(texture.index(), index);
                let img = texture.source();
                let tex = match img.source() {
                    Source::View { view, .. } => {
                        let data_buf = &buffers[view.buffer().index()].0;
                        let begin = view.offset();
                        let end = begin + view.length();
                        let bytes = &data_buf[begin..end];
                        let cursor = Cursor::new(bytes);
                        let (data, dimensions) = load_texture(cursor);
                        Texture {
                            data,
                            dimensions,
                        }
                    }
                    Source::Uri { uri, .. } => {
                        let data_url = DataUrl::parse(uri).map_err(|_| anyhow!("Failed to parse texture data"))?;
                        let bytes = data_url.get_data();
                        let cursor = Cursor::new(bytes);
                        let (data, dimensions) = load_texture(cursor);
                        Texture {
                            data,
                            dimensions,
                        }
                    }
                };
                Ok::<Texture, Error>(tex)
            })
            .collect::<Result<Vec<_>, _>>()?;
        println!("Loaded {} textures", textures.len());

        let materials = gltf.materials()
            .enumerate()
            .map(|(index, mat)| {
                assert!(mat.index().map_or(false, |i| i == index));
                let pbr = mat.pbr_metallic_roughness();

                Material {
                    index: mat.index(),
                    name: mat.name().map(Into::into),

                    base_color_factor: pbr.base_color_factor().into(),
                    base_color_texture: pbr.base_color_texture().map(|tex| tex.texture().index()),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    metallic_roughness_texture: pbr.metallic_roughness_texture().map(|tex| tex.texture().index()),

                    normal_texture: mat.normal_texture().map(|tex| tex.texture().index()),
                    normal_scale: mat.normal_texture().map(|tex| tex.scale()),

                    occlusion_texture: mat.occlusion_texture().map(|tex| tex.texture().index()),
                    occlusion_strength: mat.occlusion_texture().map(|tex| tex.strength()),

                    emissive_factor: mat.emissive_factor().into(),
                    emissive_texture: mat.emissive_texture().map(|tex| tex.texture().index()),

                    alpha_cutoff: mat.alpha_cutoff(),
                    alpha_mode: mat.alpha_mode(),

                    double_sided: mat.double_sided(),
                }
            })
            .collect::<Vec<_>>();
        println!("Loaded {} materials", materials.len());

        let meshes = gltf.meshes()
            .enumerate()
            .map(|(index, mesh)| {
                assert_eq!(mesh.index(), index);
                mesh.primitives()
                    .map(|prim| {
                        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                        let verts = reader.read_positions()
                            .ok_or(anyhow!("Unable to read positions from mesh"))?
                            .flatten()
                            .collect::<Vec<_>>();
                        let tris = reader.read_indices()
                            .ok_or(anyhow!("Unable to read indices from mesh"))?
                            .into_u32()
                            .collect::<Vec<_>>();
                        let normals = reader.read_normals()
                            .map(|normals| {
                                normals
                                    .flatten()
                                    .collect::<Vec<_>>()
                            });
                        let texcoords = reader.read_tex_coords(0)
                            .map(|texcoords| {
                                texcoords.into_f32()
                                    .flatten()
                                    .collect::<Vec<_>>()
                            });
                        Ok::<_, Error>(Mesh {
                            vertices: Self::make_buffer(verts.into_iter(), BufferUsage::VERTEX_BUFFER, memory_allocator),
                            indices: Self::make_buffer(tris.into_iter(), BufferUsage::INDEX_BUFFER, memory_allocator),
                            normals: normals.map(|normals| {
                                Self::make_buffer(
                                    normals.into_iter(),
                                    BufferUsage::VERTEX_BUFFER,
                                    memory_allocator,
                                )
                            }),
                            texcoords: texcoords.map(|texcoords| {
                                Self::make_buffer(
                                    texcoords.into_iter(),
                                    BufferUsage::VERTEX_BUFFER,
                                    memory_allocator,
                                )
                            }),
                            material: prim.material().index(),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        println!("Loaded {} primitives", meshes.iter().fold(0, |acc, v| acc + v.len()));

        let nodes = gltf.nodes()
            .enumerate()
            .map(|(index, node)| {
                assert_eq!(node.index(), index);

                (
                    node.name().unwrap_or("").to_string(),
                    Matrix4::from(node.transform().matrix()),
                    node.mesh().map(|m| m.index()),
                    node.children()
                        .map(|child| child.index())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        println!("Loaded {} nodes", nodes.len());

        let objects = {
            let mut objects = Vec::new();
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

            objects
        };
        println!("Loaded {} objects", objects.len());
        dbg!(loading_gltf_time.elapsed());
        Ok((meshes, materials, textures, objects))
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
        Buffer::from_iter(
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
