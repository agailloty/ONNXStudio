using System;
using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace ONNXStudioUI.Services;

/// <summary>
/// Decodes an image file into a CHW float32 tensor normalized to [0, 1]
/// (channel-major layout expected by vision models).
/// </summary>
public static class ImageTensorDecoder
{
    public static float[] DecodeToChw(string path, int channels, int width, int height)
    {
        if (channels is not (1 or 3 or 4) || width <= 0 || height <= 0)
            throw new ArgumentException("Expected a positive image size and 1, 3 or 4 channels.");
        using var source = new Bitmap(path);
        using var resized = new RenderTargetBitmap(new PixelSize(width, height));

        using (var context = resized.CreateDrawingContext())
        {
            context.DrawImage(source,
                new Rect(0, 0, source.PixelSize.Width, source.PixelSize.Height),
                new Rect(0, 0, width, height));
        }

        // Read BGRA pixels
        var pixels = new byte[width * height * 4];
        var handle = GCHandle.Alloc(pixels, GCHandleType.Pinned);
        try
        {
            resized.CopyPixels(new PixelRect(0, 0, width, height),
                handle.AddrOfPinnedObject(), pixels.Length, width * 4);
        }
        finally
        {
            handle.Free();
        }

        // Convert to CHW float32, normalized 0..1, RGB channel order
        var tensor = new float[channels * width * height];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixelIndex = (y * width + x) * 4;
                var offset = y * width + x;

                // RenderTargetBitmap is Bgra8888
                tensor[0 * width * height + offset] = pixels[pixelIndex + 2] / 255f; // R
                if (channels == 1)
                    tensor[offset] = (0.299f * pixels[pixelIndex + 2] + 0.587f * pixels[pixelIndex + 1] + 0.114f * pixels[pixelIndex]) / 255f;
                if (channels > 1)
                {
                    tensor[1 * width * height + offset] = pixels[pixelIndex + 1] / 255f; // G
                }
                if (channels > 2)
                {
                    tensor[2 * width * height + offset] = pixels[pixelIndex + 0] / 255f; // B
                }
                if (channels == 4) tensor[3 * width * height + offset] = pixels[pixelIndex + 3] / 255f;
            }
        }

        return tensor;
    }
}
