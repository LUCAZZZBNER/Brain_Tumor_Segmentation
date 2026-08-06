using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

public static class SegmentationV2AuditTools
{
    private sealed class Feature
    {
        public byte[] Image;
        public byte[] Mask;
        public ulong[][] Hashes;
    }

    public static string[] InspectPairs(string[] imagePaths, string[] maskPaths)
    {
        if (imagePaths.Length != maskPaths.Length)
            throw new ArgumentException("Image and mask path counts differ.");

        var output = new string[imagePaths.Length];
        for (int i = 0; i < imagePaths.Length; i++)
        {
            using (var image = new Bitmap(imagePaths[i]))
            using (var mask = new Bitmap(maskPaths[i]))
            {
                byte[] imagePixels = ReadGrayPixels(image);
                byte[] maskPixels = ReadGrayPixels(mask);
                var imageValues = new HashSet<byte>(imagePixels);
                var maskValues = new HashSet<byte>(maskPixels);
                long foreground = 0;
                int maskMinX = mask.Width, maskMinY = mask.Height, maskMaxX = -1, maskMaxY = -1;
                foreach (byte value in maskPixels) if (value >= 128) foreground++;
                for (int y = 0; y < mask.Height; y++)
                {
                    for (int x = 0; x < mask.Width; x++)
                    {
                        if (maskPixels[y * mask.Width + x] < 128) continue;
                        if (x < maskMinX) maskMinX = x;
                        if (x > maskMaxX) maskMaxX = x;
                        if (y < maskMinY) maskMinY = y;
                        if (y > maskMaxY) maskMaxY = y;
                    }
                }
                double sum = 0, sumSquared = 0;
                byte min = 255, max = 0;
                foreach (byte value in imagePixels)
                {
                    sum += value;
                    sumSquared += value * value;
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
                double mean = sum / imagePixels.Length;
                double variance = Math.Max(0.0, sumSquared / imagePixels.Length - mean * mean);
                bool binary = true;
                foreach (byte value in maskValues)
                    if (value != 0 && value != 255) binary = false;

                output[i] = Join(
                    i, image.Width, image.Height, Image.GetPixelFormatSize(image.PixelFormat),
                    mask.Width, mask.Height, Image.GetPixelFormatSize(mask.PixelFormat),
                    Sha256(imagePixels), Sha256(maskPixels), imageValues.Count, min, max,
                    mean.ToString("F6", CultureInfo.InvariantCulture),
                    Math.Sqrt(variance).ToString("F6", CultureInfo.InvariantCulture),
                    maskValues.Count, binary ? 1 : 0, foreground,
                    maskMaxX < 0 ? -1 : maskMinX, maskMaxY < 0 ? -1 : maskMinY,
                    maskMaxX, maskMaxY);
            }
        }
        return output;
    }

    public static string[] InspectRgbMaskPairs(string[] imagePaths, string[] maskPaths)
    {
        if (imagePaths.Length != maskPaths.Length)
            throw new ArgumentException("Image and mask path counts differ.");

        var output = new string[imagePaths.Length];
        for (int i = 0; i < imagePaths.Length; i++)
        {
            using (var image = new Bitmap(imagePaths[i]))
            using (var mask = new Bitmap(maskPaths[i]))
            {
                byte[] rgb = ReadRgbPixels(image);
                byte[] maskPixels = ReadGrayPixels(mask);
                bool redGreenEqual = true, redBlueEqual = true, greenBlueEqual = true;
                double[] sums = new double[3];
                double[] squared = new double[3];
                byte[] minima = { 255, 255, 255 };
                byte[] maxima = { 0, 0, 0 };
                for (int pixel = 0; pixel < rgb.Length / 3; pixel++)
                {
                    int offset = pixel * 3;
                    byte red = rgb[offset], green = rgb[offset + 1], blue = rgb[offset + 2];
                    if (red != green) redGreenEqual = false;
                    if (red != blue) redBlueEqual = false;
                    if (green != blue) greenBlueEqual = false;
                    byte[] values = { red, green, blue };
                    for (int channel = 0; channel < 3; channel++)
                    {
                        double value = values[channel];
                        sums[channel] += value;
                        squared[channel] += value * value;
                        if (values[channel] < minima[channel]) minima[channel] = values[channel];
                        if (values[channel] > maxima[channel]) maxima[channel] = values[channel];
                    }
                }
                var maskValues = new HashSet<byte>(maskPixels);
                long foreground = 0;
                foreach (byte value in maskPixels) if (value >= 128) foreground++;
                bool binary = true;
                foreach (byte value in maskValues) if (value != 0 && value != 255) binary = false;
                int count = image.Width * image.Height;
                double[] deviations = new double[3];
                for (int channel = 0; channel < 3; channel++)
                {
                    double mean = sums[channel] / count;
                    deviations[channel] = Math.Sqrt(Math.Max(0.0, squared[channel] / count - mean * mean));
                }
                output[i] = Join(
                    i, image.Width, image.Height, Image.GetPixelFormatSize(image.PixelFormat),
                    mask.Width, mask.Height, Image.GetPixelFormatSize(mask.PixelFormat),
                    Sha256(rgb), Sha256(maskPixels), maskValues.Count, binary ? 1 : 0, foreground,
                    redGreenEqual ? 1 : 0, redBlueEqual ? 1 : 0, greenBlueEqual ? 1 : 0,
                    minima[0], maxima[0], deviations[0].ToString("F6", CultureInfo.InvariantCulture),
                    minima[1], maxima[1], deviations[1].ToString("F6", CultureInfo.InvariantCulture),
                    minima[2], maxima[2], deviations[2].ToString("F6", CultureInfo.InvariantCulture));
            }
        }
        return output;
    }

    public static string[] FindCrossSplitNearDuplicates(
        string[] imagePaths, string[] maskPaths, string[] splits,
        int maximumHammingDistance, double minimumCorrelation)
    {
        if (imagePaths.Length != maskPaths.Length || imagePaths.Length != splits.Length)
            throw new ArgumentException("Path and split counts differ.");

        var features = new Feature[imagePaths.Length];
        for (int i = 0; i < imagePaths.Length; i++)
        {
            byte[] image = ResizeGray(imagePaths[i], 64, false);
            byte[] mask = ResizeGray(maskPaths[i], 64, true);
            var hashes = new ulong[8][];
            for (int transform = 0; transform < 8; transform++)
                hashes[transform] = DifferenceHash(image, 64, transform);
            features[i] = new Feature { Image = image, Mask = mask, Hashes = hashes };
        }

        var matches = new List<string>();
        for (int first = 0; first < features.Length; first++)
        {
            for (int second = first + 1; second < features.Length; second++)
            {
                if (String.Equals(splits[first], splits[second], StringComparison.OrdinalIgnoreCase))
                    continue;
                int bestDistance = Int32.MaxValue;
                int bestTransform = 0;
                double bestCorrelation = Double.NegativeInfinity;
                for (int transform = 0; transform < 8; transform++)
                {
                    int inverseTransform = InverseTransform(transform);
                    int forwardDistance = HammingDistance(features[first].Hashes[0], features[second].Hashes[transform]);
                    int reverseDistance = HammingDistance(features[second].Hashes[0], features[first].Hashes[inverseTransform]);
                    int distance = Math.Min(forwardDistance, reverseDistance);
                    if (distance > maximumHammingDistance) continue;
                    double correlation = Correlation(features[first].Image, features[second].Image, 64, transform);
                    if (correlation > bestCorrelation ||
                        (correlation == bestCorrelation && distance < bestDistance))
                    {
                        bestCorrelation = correlation;
                        bestDistance = distance;
                        bestTransform = transform;
                    }
                }
                if (bestCorrelation < minimumCorrelation) continue;
                double maskIoU = MaskIoU(features[first].Mask, features[second].Mask, 64, bestTransform);
                matches.Add(Join(
                    first, second, bestDistance,
                    bestCorrelation.ToString("F6", CultureInfo.InvariantCulture),
                    maskIoU.ToString("F6", CultureInfo.InvariantCulture),
                    TransformName(bestTransform)));
            }
        }
        return matches.ToArray();
    }

    public static string[] VerifyTransformedPairs(
        string[] firstImagePaths, string[] firstMaskPaths,
        string[] secondImagePaths, string[] secondMaskPaths, string[] transforms)
    {
        int count = firstImagePaths.Length;
        if (firstMaskPaths.Length != count || secondImagePaths.Length != count ||
            secondMaskPaths.Length != count || transforms.Length != count)
            throw new ArgumentException("Verification path counts differ.");

        var output = new string[count];
        for (int i = 0; i < count; i++)
        {
            using (var firstImage = new Bitmap(firstImagePaths[i]))
            using (var firstMask = new Bitmap(firstMaskPaths[i]))
            using (var secondImage = new Bitmap(secondImagePaths[i]))
            using (var secondMask = new Bitmap(secondMaskPaths[i]))
            {
                if (firstImage.Width != firstImage.Height || firstImage.Width != secondImage.Width ||
                    firstImage.Height != secondImage.Height || firstMask.Width != firstImage.Width ||
                    firstMask.Height != firstImage.Height || secondMask.Width != secondImage.Width ||
                    secondMask.Height != secondImage.Height)
                    throw new InvalidDataException("Full-resolution verification requires equally sized square pairs.");

                int size = firstImage.Width;
                byte[] imageA = ReadGrayPixels(firstImage);
                byte[] imageB = ReadGrayPixels(secondImage);
                byte[] maskA = ReadGrayPixels(firstMask);
                byte[] maskB = ReadGrayPixels(secondMask);
                int transform = ParseTransform(transforms[i]);
                double sumAbsolute = 0, sumSquaredError = 0;
                long imageMismatch = 0, maskMismatch = 0, intersection = 0, union = 0;
                for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                {
                    int a = imageA[y * size + x];
                    int b = At(imageB, size, x, y, transform);
                    int difference = a - b;
                    if (difference != 0) imageMismatch++;
                    sumAbsolute += Math.Abs(difference);
                    sumSquaredError += difference * difference;
                    bool ma = maskA[y * size + x] >= 128;
                    bool mb = At(maskB, size, x, y, transform) >= 128;
                    if (ma != mb) maskMismatch++;
                    if (ma && mb) intersection++;
                    if (ma || mb) union++;
                }
                double mse = sumSquaredError / imageA.Length;
                output[i] = Join(
                    i,
                    Correlation(imageA, imageB, size, transform).ToString("F9", CultureInfo.InvariantCulture),
                    (sumAbsolute / imageA.Length).ToString("F6", CultureInfo.InvariantCulture),
                    Math.Sqrt(mse).ToString("F6", CultureInfo.InvariantCulture),
                    mse == 0 ? "Infinity" : (10.0 * Math.Log10(255.0 * 255.0 / mse)).ToString("F6", CultureInfo.InvariantCulture),
                    imageMismatch,
                    (union == 0 ? 1.0 : (double)intersection / union).ToString("F9", CultureInfo.InvariantCulture),
                    maskMismatch);
            }
        }
        return output;
    }

    private static byte[] ReadGrayPixels(Bitmap bitmap)
    {
        using (var converted = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(converted))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.DrawImageUnscaled(bitmap, 0, 0);
            }
            return Read24BitGray(converted);
        }
    }

    private static byte[] ResizeGray(string path, int size, bool nearest)
    {
        using (var source = new Bitmap(path))
        using (var resized = new Bitmap(size, size, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(resized))
            {
                graphics.Clear(Color.Black);
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.InterpolationMode = nearest ? InterpolationMode.NearestNeighbor : InterpolationMode.HighQualityBilinear;
                graphics.PixelOffsetMode = nearest ? PixelOffsetMode.Half : PixelOffsetMode.HighQuality;
                graphics.DrawImage(source, new Rectangle(0, 0, size, size));
            }
            byte[] values = Read24BitGray(resized);
            if (nearest)
                for (int i = 0; i < values.Length; i++) values[i] = values[i] >= 128 ? (byte)255 : (byte)0;
            return values;
        }
    }

    private static byte[] Read24BitGray(Bitmap bitmap)
    {
        var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            int stride = Math.Abs(data.Stride);
            byte[] raw = new byte[stride * bitmap.Height];
            Marshal.Copy(data.Scan0, raw, 0, raw.Length);
            byte[] output = new byte[bitmap.Width * bitmap.Height];
            for (int y = 0; y < bitmap.Height; y++)
            {
                int row = (data.Stride > 0 ? y : bitmap.Height - 1 - y) * stride;
                for (int x = 0; x < bitmap.Width; x++)
                {
                    int offset = row + x * 3;
                    output[y * bitmap.Width + x] = (byte)((raw[offset + 2] * 299 + raw[offset + 1] * 587 + raw[offset] * 114 + 500) / 1000);
                }
            }
            return output;
        }
        finally { bitmap.UnlockBits(data); }
    }

    private static byte[] ReadRgbPixels(Bitmap bitmap)
    {
        using (var converted = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(converted))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.DrawImageUnscaled(bitmap, 0, 0);
            }
            var rectangle = new Rectangle(0, 0, converted.Width, converted.Height);
            BitmapData data = converted.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                int stride = Math.Abs(data.Stride);
                byte[] raw = new byte[stride * converted.Height];
                Marshal.Copy(data.Scan0, raw, 0, raw.Length);
                byte[] output = new byte[converted.Width * converted.Height * 3];
                for (int y = 0; y < converted.Height; y++)
                {
                    int row = (data.Stride > 0 ? y : converted.Height - 1 - y) * stride;
                    for (int x = 0; x < converted.Width; x++)
                    {
                        int source = row + x * 3;
                        int target = (y * converted.Width + x) * 3;
                        output[target] = raw[source + 2];
                        output[target + 1] = raw[source + 1];
                        output[target + 2] = raw[source];
                    }
                }
                return output;
            }
            finally { converted.UnlockBits(data); }
        }
    }

    private static ulong[] DifferenceHash(byte[] pixels, int size, int transform)
    {
        var hash = new ulong[4];
        int bit = 0;
        for (int y = 0; y < 16; y++)
        {
            int sampleY = Math.Min(size - 1, y * size / 16 + size / 32);
            for (int x = 0; x < 16; x++)
            {
                int sampleX1 = Math.Min(size - 1, x * size / 17 + size / 34);
                int sampleX2 = Math.Min(size - 1, (x + 1) * size / 17 + size / 34);
                if (At(pixels, size, sampleX1, sampleY, transform) > At(pixels, size, sampleX2, sampleY, transform))
                    hash[bit / 64] |= 1UL << (bit % 64);
                bit++;
            }
        }
        return hash;
    }

    private static double Correlation(byte[] first, byte[] second, int size, int transform)
    {
        double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
        int count = size * size;
        for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
        {
            double a = first[y * size + x];
            double b = At(second, size, x, y, transform);
            sumA += a; sumB += b; sumAA += a * a; sumBB += b * b; sumAB += a * b;
        }
        double denominator = Math.Sqrt((count * sumAA - sumA * sumA) * (count * sumBB - sumB * sumB));
        return denominator == 0 ? 0 : (count * sumAB - sumA * sumB) / denominator;
    }

    private static double MaskIoU(byte[] first, byte[] second, int size, int transform)
    {
        int intersection = 0, union = 0;
        for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
        {
            bool a = first[y * size + x] >= 128;
            bool b = At(second, size, x, y, transform) >= 128;
            if (a && b) intersection++;
            if (a || b) union++;
        }
        return union == 0 ? 1.0 : (double)intersection / union;
    }

    private static byte At(byte[] pixels, int size, int x, int y, int transform)
    {
        int sourceX, sourceY;
        switch (transform)
        {
            case 1: sourceX = size - 1 - x; sourceY = y; break;                  // flip horizontal
            case 2: sourceX = x; sourceY = size - 1 - y; break;                  // flip vertical
            case 3: sourceX = size - 1 - x; sourceY = size - 1 - y; break;       // rotate 180
            case 4: sourceX = y; sourceY = size - 1 - x; break;                  // rotate 90
            case 5: sourceX = size - 1 - y; sourceY = x; break;                  // rotate 270
            case 6: sourceX = y; sourceY = x; break;                              // transpose
            case 7: sourceX = size - 1 - y; sourceY = size - 1 - x; break;       // anti-transpose
            default: sourceX = x; sourceY = y; break;
        }
        return pixels[sourceY * size + sourceX];
    }

    private static string TransformName(int transform)
    {
        string[] names = { "identity", "flip_h", "flip_v", "rotate_180", "rotate_90", "rotate_270", "transpose", "anti_transpose" };
        return names[transform];
    }

    private static int ParseTransform(string name)
    {
        string[] names = { "identity", "flip_h", "flip_v", "rotate_180", "rotate_90", "rotate_270", "transpose", "anti_transpose" };
        for (int i = 0; i < names.Length; i++)
            if (String.Equals(name, names[i], StringComparison.OrdinalIgnoreCase)) return i;
        throw new ArgumentException("Unknown transform: " + name);
    }

    private static int InverseTransform(int transform)
    {
        if (transform == 4) return 5;
        if (transform == 5) return 4;
        return transform;
    }

    private static int HammingDistance(ulong[] first, ulong[] second)
    {
        int count = 0;
        for (int i = 0; i < first.Length; i++)
        {
            ulong value = first[i] ^ second[i];
            while (value != 0) { value &= value - 1; count++; }
        }
        return count;
    }

    private static string Sha256(byte[] bytes)
    {
        using (SHA256 sha = SHA256.Create())
        {
            var builder = new StringBuilder(64);
            foreach (byte value in sha.ComputeHash(bytes)) builder.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            return builder.ToString();
        }
    }

    private static string Join(params object[] values)
    {
        return String.Join("|", Array.ConvertAll(values, value => Convert.ToString(value, CultureInfo.InvariantCulture)));
    }
}
