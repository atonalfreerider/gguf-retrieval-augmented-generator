using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using LLama;
using LLama.Common;

namespace gguf_RAG;

static class Program
{
    class Args
    {
        public string InputModelPath { get; set; }
        public string VectorDbPath { get; set; }
        public string? TrainingDataFolderPath { get; set; }
    }

    static void Main(string[] args)
    {
        RootCommand rootCommand = new()
        {
            new Argument<string>(
                "InputModelPath",
                "Input mode path. Must be .gguf format. Examples: https://huggingface.co/TheBloke"),

            new Argument<string>(
                "VectorDbPath",
                "Previously trained, or new file containing vector data."),

            new Option<string>(
                "--TrainingDataFolderPath",
                "Path to a folder containing text files to train on.")
        };

        rootCommand.Description = "Train a Vector db on data and chat with LLM.";

        // Note that the parameters of the handler method are matched according to the names of the options 
        rootCommand.Handler = CommandHandler.Create<Args>(Parse);

        rootCommand.Invoke(args);

        Environment.Exit(0);
    }

    static async void Parse(Args args)
    {
        if (!args.InputModelPath.EndsWith(".gguf"))
        {
            Console.WriteLine("model must be in .gguf format");
            Environment.Exit(1);
        }

        // Load a model
        ModelParams parameters = new ModelParams(args.InputModelPath)
        {
            ContextSize = 1024,
            MainGpu = 0,
            EmbeddingMode = true
        };
        using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);
        using LLamaContext context = model.CreateContext(parameters);
        using LLamaEmbedder embedder = new LLamaEmbedder(model, context.Params);
        VectorDb vectorDb = new VectorDb(embedder);

        if (!File.Exists(args.VectorDbPath) &&
            !string.IsNullOrEmpty(args.TrainingDataFolderPath) &&
            Directory.Exists(args.TrainingDataFolderPath))
        {
            List<string> trainingData = await DocReader.ReadFolder(args.TrainingDataFolderPath, (int)parameters.ContextSize);
            
            vectorDb.Train(trainingData);

            SqliteOutput sqliteOutput = new SqliteOutput(args.VectorDbPath);
            sqliteOutput.Serialize(vectorDb.VectorsByChunk);
            
            Console.WriteLine($"Embedded data saved to {args.VectorDbPath}");
        }
        else if (File.Exists(args.VectorDbPath))
        {
            vectorDb.VectorsByChunk = SqliteInput.ReadFromVectorDb(args.VectorDbPath, embedder.EmbeddingSize);
        }
        else
        {
            Console.WriteLine("Vector db not found");
            Environment.Exit(1);
        }

        // Initialize a chat session
        InteractiveExecutor ex = new InteractiveExecutor(context);
        ChatSession session = new ChatSession(ex);

        Console.Clear();
        
        // run the inference in a loop to chat with LLM
        string? prompt = "Enter prompt";
        Console.WriteLine(prompt);
        while (prompt != "exit")
        {
            prompt = Console.ReadLine();
            if(string.IsNullOrEmpty(prompt)) continue;
            float[] embeddings = embedder.GetEmbeddings(prompt);
            string topText = vectorDb.TextFromEmbedding(embeddings, 3);
            prompt = $"Using the text passages below, please answer the user's question in the voice of the author:\n\n {topText} \n\n Question: {prompt}";
            
            await foreach (string text in session.ChatAsync(
                               prompt,
                               new InferenceParams { Temperature = 0.6f }))
            {
                Console.Write(text);
            }
        }
    }
}