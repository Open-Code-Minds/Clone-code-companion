const Jimp = require('jimp');
const MAX_IMAGE_DIMENSION = 1280;

class ImageHandler {
  constructor() {}

  async imageToBase64(filePath) {
    const imageBuffer = await this.resizeImageIfNeeded(filePath);
    const mimeType = imageBuffer.getMIME();
    const base64Image = await imageBuffer.getBase64Async(mimeType);

    return { base64Image, mimeType };
  }

  async resizeImageIfNeeded(filePath) {
    const image = await Jimp.read(filePath);
    const { width, height } = image.bitmap;

    if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
      const aspectRatio = width / height;
      if (aspectRatio > 1) {
        return image.resize(MAX_IMAGE_DIMENSION, Jimp.AUTO).quality(90);
      } else {
        return image.resize(Jimp.AUTO, MAX_IMAGE_DIMENSION).quality(90);
      }
    } else {
      return image;
    }
  }
}

module.exports = ImageHandler;
