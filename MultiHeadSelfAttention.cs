using System;
using ICSharpCode.SharpZipLib.Zip.Compression.Streams;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BerdyGPT
{
    public class MultiHeadSelfAttention : Module<Tensor, Tensor>
    {
        private readonly Linear wQ, wK, wV, wO;
        private readonly RotaryPositionalEmbedding rope;
        private int EmbeddingSize, HeadsCount, HeadSize;

        public MultiHeadSelfAttention(int embeddingSize, int headsCount) :
        base(nameof(MultiHeadSelfAttention))
        {
            if (embeddingSize % headsCount != 0)
                throw new Exception("E % H != 0");

            EmbeddingSize = embeddingSize;
            HeadsCount = headsCount;

            HeadSize = EmbeddingSize / HeadsCount;

            rope = new RotaryPositionalEmbedding(HeadSize, 512);

            wQ = Linear(EmbeddingSize, EmbeddingSize);
            wK = Linear(EmbeddingSize, EmbeddingSize);
            wV = Linear(EmbeddingSize, EmbeddingSize);

            wO = Linear(EmbeddingSize, EmbeddingSize, hasBias: false);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            bool isUnbatched = input.dim() == 2;

            if (isUnbatched) input.unsqueeze(0);

            long b = input.shape[0];
            long t = input.shape[1];

            var q = wQ.forward(input).view(b, t, HeadsCount, HeadSize).transpose(1, 2);
            var k = wK.forward(input).view(b, t, HeadsCount, HeadSize).transpose(1, 2);
            var v = wV.forward(input).view(b, t, HeadsCount, HeadSize).transpose(1, 2);

            q = rope.Apply(q);
            k = rope.Apply(k);

            var scores = matmul(q, k.transpose(-1, -2)) / MathF.Sqrt(HeadSize);

            var casualMask = triu(ones(new long[] { t, t }, ScalarType.Bool, device: input.device), diagonal: 1);
            scores = scores.masked_fill(casualMask, float.NegativeInfinity);

            var attention = functional.softmax(scores, 1);

            var headOutput = matmul(attention, v);

            var concat = headOutput.transpose(1, 2).contiguous().view(b, t, EmbeddingSize);

            var output = wO.forward(concat);

            return isUnbatched ? output.unsqueeze(0) : output;
        }
    }
}