using System;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BerdyGPT
{
    public sealed class RotaryPositionalEmbedding : IDisposable
    {
        private readonly Tensor cosCache;
        private readonly Tensor sinCache;

        private readonly int headDim;
        private readonly int maxSeqLen;
        private bool disposed;

        public RotaryPositionalEmbedding(
            int headDim,
            int maxSeqLen,
            float theta = 10000f)
        {
            if (headDim <= 0 || headDim % 2 != 0)
                throw new ArgumentException("Head dimension must be positive and even.");

            if (maxSeqLen <= 0)
                throw new ArgumentException("Maximum sequence length must be positive.");

            if (theta <= 0)
                throw new ArgumentException("Theta must be positive.");

            this.headDim = headDim;
            this.maxSeqLen = maxSeqLen;

            int pairs = headDim / 2;

            float[] cosValues = new float[maxSeqLen * pairs];
            float[] sinValues = new float[maxSeqLen * pairs];

            for (int pos = 0; pos < maxSeqLen; pos++)
            {
                for (int i = 0; i < pairs; i++)
                {
                    double frequency = Math.Pow(theta, -(2.0 * i / headDim));
                    double angle = pos * frequency;

                    int index = pos * pairs + i;

                    cosValues[index] = (float)Math.Cos(angle);
                    sinValues[index] = (float)Math.Sin(angle);
                }
            }

            cosCache = torch.tensor(cosValues, dtype: ScalarType.Float32)
                .reshape(maxSeqLen, pairs);

            sinCache = torch.tensor(sinValues, dtype: ScalarType.Float32)
                .reshape(maxSeqLen, pairs);
        }

        public Tensor Apply(Tensor x, int offset = 0)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (x.shape[^1] != headDim) throw new ArgumentException("Input head dimension does not match RoPE.");

            int seqLen = checked((int)x.shape[^2]);

            if (offset < 0 || (long)offset + seqLen > maxSeqLen) throw new ArgumentOutOfRangeException(nameof(offset));

            using var cos = cosCache
                .slice(0, offset, offset + seqLen, 1)
                .to(x.device)
                .unsqueeze(0)
                .unsqueeze(0);

            using var sin = sinCache
                .slice(0, offset, offset + seqLen, 1)
                .to(x.device)
                .unsqueeze(0)
                .unsqueeze(0);

            using var even = x.slice(-1, 0, headDim, 2);
            using var odd = x.slice(-1, 1, headDim, 2);

            using var rotatedEven = even * cos - odd * sin;
            using var rotatedOdd = even * sin + odd * cos;

            using var stacked = torch.stack(
                new[] { rotatedEven, rotatedOdd },
                dim: -1);

            return stacked.flatten(-2);
        }

        public void Dispose()
        {
            if (disposed)
                return;

            cosCache.Dispose();
            sinCache.Dispose();

            disposed = true;
        }
    }
}