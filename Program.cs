namespace BerdyGPT;

class Program
{
    private static readonly Tokenizer tokenizer = new Tokenizer();

    static void Main(string[] args)
    {
        Console.Write(">");

        string? input = Console.ReadLine();

        IReadOnlyCollection<int> tokensIds = tokenizer.Encode(input!);

        foreach (var tokenId in tokensIds)
        {
            Console.Write(string.Join(",", tokenId));
        }

        Console.WriteLine(tokenizer.Decode(tokensIds));
    }
}
