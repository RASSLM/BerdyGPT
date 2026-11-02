using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BerdyGPT
{
    public class FeedForwardNetwork : Module<Tensor, Tensor>
    {
        private readonly int EmbeddingSize, FFN_Size;
        private Linear linear1, linear2;

        public FeedForwardNetwork(int embeddingSize, int ffnSize) : base(nameof(FeedForwardNetwork))
        {
            EmbeddingSize = embeddingSize;
            FFN_Size = ffnSize;

            linear1 = Linear(EmbeddingSize, FFN_Size);
            linear2 = Linear(FFN_Size, EmbeddingSize);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = linear1.forward(input);

            var hidden = 0.5f * x * (1f + torch.tanh(MathF.Sqrt(2f / MathF.PI)
             * (x + 0.044715f * torch.pow(x, 3)
            )));

            return linear2.forward(hidden);
        }
    }
}