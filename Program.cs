using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using LLama;
using LLama.Common;

static class Program
{
    class Args
    {
        public string InputPath { get; set; }
    }

    static void Main(string[] args)
    {
        RootCommand rootCommand = new()
        {
            new Argument<string>(
                "InputPath",
                "Input path"),
        };

        rootCommand.Description = "Train a Vector db";

        // Note that the parameters of the handler method are matched according to the names of the options 
        rootCommand.Handler = CommandHandler.Create<Args>(Parse);

        rootCommand.Invoke(args);

        Environment.Exit(0);
    }

    static async void Parse(Args args)
    {
        string modelPath = @"C:\Users\johnb\Desktop\llama-models\llama-2-7b-guanaco-qlora.Q8_0.gguf";
        string? prompt =
            "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest," +
            " good at writing, and never fails to answer the User's requests immediately and with precision." +
            "\r\n\r\nUser: Hello, Bob." +
            "\r\nBob: Hello. How may I help you today?" +
            "\r\nUser: Please tell me the largest city in Europe." +
            "\r\nBob: Sure. The largest city in Europe is Moscow, the capital of Russia." +
            "\r\nUser:"; // use the "chat-with-bob" prompt here.

        // Load a model
        ModelParams parameters = new ModelParams(modelPath)
        {
            ContextSize = 1024,
            Seed = 1337,
            GpuLayerCount = 5
        };
        using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);

        // Initialize a chat session
        using LLamaContext context = model.CreateContext(parameters);
        InteractiveExecutor ex = new InteractiveExecutor(context);
        ChatSession session = new ChatSession(ex);

        // show the prompt
        Console.WriteLine();
        Console.Write(prompt);

        // run the inference in a loop to chat with LLM
        while (prompt != "stop")
        {
            await foreach (string text in session.ChatAsync(prompt,
                               new InferenceParams() { Temperature = 0.6f, AntiPrompts = new List<string> { "User:" } }))
            {
                Console.Write(text);
            }

            prompt = Console.ReadLine();
        }

        // save the session
        session.SaveSession("SavedSessionPath");
    }
}