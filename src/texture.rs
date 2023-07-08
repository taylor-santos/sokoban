use std::io::{BufRead, Read, Seek};

use image::{
    GenericImageView,
    io::Reader,
    RgbaImage,
};
use vulkano::image::ImageDimensions;

pub fn load_texture<R: Read + BufRead + Seek>(
    cursor: R
) -> (RgbaImage, ImageDimensions) {
    let reader = Reader::new(cursor)
        .with_guessed_format()
        .expect("Unable to read image file");
    let img = reader
        .decode()
        .expect("Unable to decode image");

    let (width, height) = img.dimensions();

    let dimensions = ImageDimensions::Dim2d {
        width,
        height,
        array_layers: 1,
    };

    (img.into_rgba8(), dimensions)
}
