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

public sealed class V2PairStats
{
    public int Width;
    public int Height;
    public long ForegroundPixels;
    public long TotalPixels;
    public int ImageDistinctValues;
    public int MaskDistinctValues;

    public double ForegroundFraction
    {
        get { return TotalPixels == 0 ? 0.0 : (double)ForegroundPixels / TotalPixels; }
    }
}

public static class V2ImageTools
{
    public static string Sha256File(string path)
    {
        using (FileStream stream = File.OpenRead(path))
        using (SHA256 sha = SHA256.Create())
        {
            return ToHex(sha.ComputeHash(stream));
        }
    }

    public static string Sha256Text(string value)
    {
        using (SHA256 sha = SHA256.Create())
        {
            return ToHex(sha.ComputeHash(Encoding.UTF8.GetBytes(value)));
        }
    }

    public static string[] FindCrossDatasetNearDuplicates(
        string[] leftPaths,
        string[] rightPaths,
        int maxHammingDistance,
        double minimumCorrelation)
    {
        ulong[][] leftHashes = new ulong[leftPaths.Length][];
        ulong[][] rightHashes = new ulong[rightPaths.Length][];
        for (int index = 0; index < leftPaths.Length; index++)
        {
            leftHashes[index] = DifferenceHash(leftPaths[index]);
        }
        for (int index = 0; index < rightPaths.Length; index++)
        {
            rightHashes[index] = DifferenceHash(rightPaths[index]);
        }

        List<string> matches = new List<string>();
        for (int left = 0; left < leftHashes.Length; left++)
        {
            for (int right = 0; right < rightHashes.Length; right++)
            {
                int distance = HammingDistance(leftHashes[left], rightHashes[right]);
                if (distance > maxHammingDistance)
                {
                    continue;
                }
                double correlation = PixelCorrelation(leftPaths[left], rightPaths[right]);
                if (correlation >= minimumCorrelation)
                {
                    matches.Add(
                        left.ToString(CultureInfo.InvariantCulture) + "," +
                        right.ToString(CultureInfo.InvariantCulture) + "," +
                        distance.ToString(CultureInfo.InvariantCulture) + "," +
                        correlation.ToString("F6", CultureInfo.InvariantCulture));
                }
            }
        }
        return matches.ToArray();
    }

    public static string[] FindWithinDatasetNearDuplicates(
        string[] paths,
        int maxHammingDistance,
        double minimumCorrelation)
    {
        ulong[][] hashes = new ulong[paths.Length][];
        for (int index = 0; index < paths.Length; index++)
        {
            hashes[index] = DifferenceHash(paths[index]);
        }

        List<string> matches = new List<string>();
        for (int first = 0; first < hashes.Length; first++)
        {
            for (int second = first + 1; second < hashes.Length; second++)
            {
                int distance = HammingDistance(hashes[first], hashes[second]);
                if (distance > maxHammingDistance)
                {
                    continue;
                }
                double correlation = PixelCorrelation(paths[first], paths[second]);
                if (correlation >= minimumCorrelation)
                {
                    matches.Add(
                        first.ToString(CultureInfo.InvariantCulture) + "," +
                        second.ToString(CultureInfo.InvariantCulture) + "," +
                        distance.ToString(CultureInfo.InvariantCulture) + "," +
                        correlation.ToString("F6", CultureInfo.InvariantCulture));
                }
            }
        }
        return matches.ToArray();
    }

    public static double NormalizedMaskIoU(string firstPath, string secondPath, int targetSize)
    {
        byte[] first = NormalizedMask(firstPath, targetSize);
        byte[] second = NormalizedMask(secondPath, targetSize);
        long intersection = 0;
        long union = 0;
        for (int index = 0; index < first.Length; index++)
        {
            bool a = first[index] != 0;
            bool b = second[index] != 0;
            if (a && b)
            {
                intersection++;
            }
            if (a || b)
            {
                union++;
            }
        }
        return union == 0 ? 1.0 : (double)intersection / union;
    }

    public static V2PairStats ConvertPair(
        string imagePath,
        string maskPath,
        string outputImagePath,
        string outputMaskPath,
        int targetSize)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputImagePath));
        Directory.CreateDirectory(Path.GetDirectoryName(outputMaskPath));

        using (Bitmap image = new Bitmap(imagePath))
        using (Bitmap mask = new Bitmap(maskPath))
        {
            if (image.Width != mask.Width || image.Height != mask.Height)
            {
                throw new InvalidDataException("Image/mask size mismatch: " + imagePath);
            }

            using (Bitmap normalizedImage = NormalizeImage(image, targetSize))
            using (Bitmap normalizedMask = NormalizeMaskBitmap(mask, targetSize))
            {
                normalizedImage.Save(outputImagePath, ImageFormat.Png);
                normalizedMask.Save(outputMaskPath, ImageFormat.Png);
            }
        }
        return InspectPair(outputImagePath, outputMaskPath, targetSize);
    }

    public static V2PairStats InspectPair(string imagePath, string maskPath, int targetSize)
    {
        using (Bitmap image = new Bitmap(imagePath))
        using (Bitmap mask = new Bitmap(maskPath))
        {
            if (image.Width != targetSize || image.Height != targetSize ||
                mask.Width != targetSize || mask.Height != targetSize)
            {
                throw new InvalidDataException("Normalized pair has an unexpected size: " + imagePath);
            }
            if (Image.GetPixelFormatSize(image.PixelFormat) != 8 ||
                Image.GetPixelFormatSize(mask.PixelFormat) != 8)
            {
                throw new InvalidDataException("Normalized pair is not 8-bit grayscale: " + imagePath);
            }

            int imageDistinct;
            long ignoredForeground;
            ReadIndexedBitmap(image, false, out imageDistinct, out ignoredForeground);
            int maskDistinct;
            long foreground;
            HashSet<int> maskValues = ReadIndexedBitmap(mask, true, out maskDistinct, out foreground);
            foreach (int value in maskValues)
            {
                if (value != 0 && value != 255)
                {
                    throw new InvalidDataException("Normalized mask is not binary: " + maskPath);
                }
            }
            long pixels = (long)targetSize * targetSize;
            if (foreground == 0 || foreground == pixels)
            {
                throw new InvalidDataException("Normalized mask lacks foreground or background: " + maskPath);
            }
            return new V2PairStats
            {
                Width = targetSize,
                Height = targetSize,
                ForegroundPixels = foreground,
                TotalPixels = pixels,
                ImageDistinctValues = imageDistinct,
                MaskDistinctValues = maskDistinct
            };
        }
    }

    private static Bitmap NormalizeImage(Bitmap source, int targetSize)
    {
        int width;
        int height;
        int offsetX;
        int offsetY;
        LetterboxGeometry(source.Width, source.Height, targetSize, out width, out height, out offsetX, out offsetY);

        using (Bitmap resized = new Bitmap(width, height, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(resized))
            {
                graphics.Clear(Color.Black);
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                graphics.SmoothingMode = SmoothingMode.None;
                graphics.DrawImage(source, new Rectangle(0, 0, width, height));
            }
            Bitmap output = CreateGrayscaleBitmap(targetSize, targetSize);
            CopyResizedGrayscale(resized, output, offsetX, offsetY, false);
            return output;
        }
    }

    private static Bitmap NormalizeMaskBitmap(Bitmap source, int targetSize)
    {
        int width;
        int height;
        int offsetX;
        int offsetY;
        LetterboxGeometry(source.Width, source.Height, targetSize, out width, out height, out offsetX, out offsetY);

        using (Bitmap resized = new Bitmap(width, height, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(resized))
            {
                graphics.Clear(Color.Black);
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighSpeed;
                graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                graphics.PixelOffsetMode = PixelOffsetMode.Half;
                graphics.SmoothingMode = SmoothingMode.None;
                graphics.DrawImage(source, new Rectangle(0, 0, width, height));
            }
            Bitmap output = CreateGrayscaleBitmap(targetSize, targetSize);
            CopyResizedGrayscale(resized, output, offsetX, offsetY, true);
            return output;
        }
    }

    private static byte[] NormalizedMask(string path, int targetSize)
    {
        using (Bitmap source = new Bitmap(path))
        using (Bitmap normalized = NormalizeMaskBitmap(source, targetSize))
        {
            Rectangle rectangle = new Rectangle(0, 0, targetSize, targetSize);
            BitmapData data = normalized.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
            try
            {
                int stride = Math.Abs(data.Stride);
                byte[] raw = new byte[stride * targetSize];
                Marshal.Copy(data.Scan0, raw, 0, raw.Length);
                byte[] result = new byte[targetSize * targetSize];
                for (int y = 0; y < targetSize; y++)
                {
                    int sourceRow = (data.Stride > 0 ? y : targetSize - 1 - y) * stride;
                    Buffer.BlockCopy(raw, sourceRow, result, y * targetSize, targetSize);
                }
                return result;
            }
            finally
            {
                normalized.UnlockBits(data);
            }
        }
    }

    private static void LetterboxGeometry(
        int sourceWidth,
        int sourceHeight,
        int targetSize,
        out int width,
        out int height,
        out int offsetX,
        out int offsetY)
    {
        double scale = Math.Min((double)targetSize / sourceWidth, (double)targetSize / sourceHeight);
        width = Math.Max(1, Math.Min(targetSize, (int)Math.Round(sourceWidth * scale)));
        height = Math.Max(1, Math.Min(targetSize, (int)Math.Round(sourceHeight * scale)));
        offsetX = (targetSize - width) / 2;
        offsetY = (targetSize - height) / 2;
    }

    private static Bitmap CreateGrayscaleBitmap(int width, int height)
    {
        Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
        ColorPalette palette = bitmap.Palette;
        for (int index = 0; index < 256; index++)
        {
            palette.Entries[index] = Color.FromArgb(index, index, index);
        }
        bitmap.Palette = palette;
        return bitmap;
    }

    private static void CopyResizedGrayscale(
        Bitmap resized,
        Bitmap output,
        int offsetX,
        int offsetY,
        bool threshold)
    {
        Rectangle sourceRectangle = new Rectangle(0, 0, resized.Width, resized.Height);
        BitmapData sourceData = resized.LockBits(sourceRectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        Rectangle outputRectangle = new Rectangle(0, 0, output.Width, output.Height);
        BitmapData outputData = output.LockBits(outputRectangle, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
        try
        {
            int sourceStride = Math.Abs(sourceData.Stride);
            int outputStride = Math.Abs(outputData.Stride);
            byte[] sourceBytes = new byte[sourceStride * resized.Height];
            byte[] outputBytes = new byte[outputStride * output.Height];
            Marshal.Copy(sourceData.Scan0, sourceBytes, 0, sourceBytes.Length);
            for (int y = 0; y < resized.Height; y++)
            {
                int sourceRow = (sourceData.Stride > 0 ? y : resized.Height - 1 - y) * sourceStride;
                int outputRow = (outputData.Stride > 0 ? y + offsetY : output.Height - 1 - (y + offsetY)) * outputStride;
                for (int x = 0; x < resized.Width; x++)
                {
                    int sourceOffset = sourceRow + x * 3;
                    int blue = sourceBytes[sourceOffset];
                    int green = sourceBytes[sourceOffset + 1];
                    int red = sourceBytes[sourceOffset + 2];
                    int gray = (red * 299 + green * 587 + blue * 114 + 500) / 1000;
                    outputBytes[outputRow + x + offsetX] = (byte)(threshold ? (gray >= 128 ? 255 : 0) : gray);
                }
            }
            Marshal.Copy(outputBytes, 0, outputData.Scan0, outputBytes.Length);
        }
        finally
        {
            resized.UnlockBits(sourceData);
            output.UnlockBits(outputData);
        }
    }

    private static HashSet<int> ReadIndexedBitmap(
        Bitmap bitmap,
        bool countForeground,
        out int distinct,
        out long foreground)
    {
        Rectangle rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
        try
        {
            int stride = Math.Abs(data.Stride);
            byte[] raw = new byte[stride * bitmap.Height];
            Marshal.Copy(data.Scan0, raw, 0, raw.Length);
            ColorPalette palette = bitmap.Palette;
            HashSet<int> values = new HashSet<int>();
            foreground = 0;
            for (int y = 0; y < bitmap.Height; y++)
            {
                int row = (data.Stride > 0 ? y : bitmap.Height - 1 - y) * stride;
                for (int x = 0; x < bitmap.Width; x++)
                {
                    int value = palette.Entries[raw[row + x]].R;
                    values.Add(value);
                    if (countForeground && value >= 128)
                    {
                        foreground++;
                    }
                }
            }
            distinct = values.Count;
            return values;
        }
        finally
        {
            bitmap.UnlockBits(data);
        }
    }

    private static ulong[] DifferenceHash(string path)
    {
        using (Bitmap source = new Bitmap(path))
        using (Bitmap resized = new Bitmap(17, 16, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(resized))
            {
                graphics.Clear(Color.Black);
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.DrawImage(source, 0, 0, 17, 16);
            }
            ulong[] hash = new ulong[4];
            int bit = 0;
            for (int y = 0; y < 16; y++)
            {
                for (int x = 0; x < 16; x++)
                {
                    int first = Gray(resized.GetPixel(x, y));
                    int second = Gray(resized.GetPixel(x + 1, y));
                    if (first > second)
                    {
                        hash[bit / 64] |= 1UL << (bit % 64);
                    }
                    bit++;
                }
            }
            return hash;
        }
    }

    private static double PixelCorrelation(string firstPath, string secondPath)
    {
        using (Bitmap firstSource = new Bitmap(firstPath))
        using (Bitmap secondSource = new Bitmap(secondPath))
        using (Bitmap first = new Bitmap(64, 64, PixelFormat.Format24bppRgb))
        using (Bitmap second = new Bitmap(64, 64, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(first))
            {
                graphics.Clear(Color.Black);
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.DrawImage(firstSource, 0, 0, 64, 64);
            }
            using (Graphics graphics = Graphics.FromImage(second))
            {
                graphics.Clear(Color.Black);
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.DrawImage(secondSource, 0, 0, 64, 64);
            }

            double sumFirst = 0;
            double sumSecond = 0;
            double sumFirstSquared = 0;
            double sumSecondSquared = 0;
            double sumProduct = 0;
            const int count = 64 * 64;
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    double a = Gray(first.GetPixel(x, y));
                    double b = Gray(second.GetPixel(x, y));
                    sumFirst += a;
                    sumSecond += b;
                    sumFirstSquared += a * a;
                    sumSecondSquared += b * b;
                    sumProduct += a * b;
                }
            }
            double denominator = Math.Sqrt(
                (count * sumFirstSquared - sumFirst * sumFirst) *
                (count * sumSecondSquared - sumSecond * sumSecond));
            return denominator == 0 ? 0.0 :
                (count * sumProduct - sumFirst * sumSecond) / denominator;
        }
    }

    private static int Gray(Color color)
    {
        return (color.R * 299 + color.G * 587 + color.B * 114 + 500) / 1000;
    }

    private static int HammingDistance(ulong[] first, ulong[] second)
    {
        return PopulationCount(first[0] ^ second[0]) +
               PopulationCount(first[1] ^ second[1]) +
               PopulationCount(first[2] ^ second[2]) +
               PopulationCount(first[3] ^ second[3]);
    }

    private static int PopulationCount(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            value &= value - 1;
            count++;
        }
        return count;
    }

    private static string ToHex(byte[] bytes)
    {
        StringBuilder builder = new StringBuilder(bytes.Length * 2);
        foreach (byte value in bytes)
        {
            builder.Append(value.ToString("x2", CultureInfo.InvariantCulture));
        }
        return builder.ToString();
    }
}
