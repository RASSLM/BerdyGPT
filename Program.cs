using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BerdyGPT;

class Program
{
    private static readonly Tokenizer tokenizer = new Tokenizer();
    private static readonly Tokenizer tokenizer_ = new Tokenizer();
    private static LanguageModel model;

    static void Main(string[] args)
    {
        model = new LanguageModel(128, 4, 512, 3);

        Console.Write("=>");

        string? input = Console.ReadLine();

        Console.WriteLine();

        var prompt = tokenizer.Encode(input!);

        var promptTokens = prompt.ToArray();

        var device = cuda.is_available() ? CUDA : CPU;

        var replyTokens = model.Generate(promptTokens, 12, 15465, device);

        string reply = tokenizer_.Decode(replyTokens);

        Console.WriteLine(reply);
    }
}
