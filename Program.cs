using System.Collections.Immutable;
using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using LLama;
using LLama.Common;
using PdfSharp.Pdf;
using PdfSharp.Pdf.IO;

namespace vector_db;

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
                "Input mode path. Must be .gguf format. Example https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GGUF/tree/main"),

            new Argument<string>(
                "VectorDbPath",
                "Previously trained, or new file containing author data."),

            new Option<string>(
                "--TrainingDataFolderPath",
                "Path to a folder containing text files to train on.")
        };

        rootCommand.Description = "Train a Vector db on a body of work and chat with the author";

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
            Seed = 1337,
            GpuLayerCount = 5
        };
        using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);
        using LLamaContext context = model.CreateContext(parameters);
        using LLamaEmbedder embedder = new LLamaEmbedder(model, context.Params);
        Dictionary<string, float[]> vectorDb = new();

        if (!File.Exists(args.VectorDbPath) &&
            !string.IsNullOrEmpty(args.TrainingDataFolderPath) &&
            Directory.Exists(args.TrainingDataFolderPath))
        {
            List<string> trainingData = new();
            foreach (string file in Directory.EnumerateFiles(args.TrainingDataFolderPath))
            {
                Console.WriteLine($"reading {file}");
               
                if (file.EndsWith(".pdf"))
                {
                    PdfDocument pdfDocument = PdfReader.Open(file);
                    foreach (PdfPage page in pdfDocument.Pages)
                    {
                        string agg = string.Join("", page.ExtractText());
                        trainingData.AddRange(Chunk(agg, (int)parameters.ContextSize));
                    }
                }
                else if (file.EndsWith(".txt"))
                {
                    trainingData.AddRange(Chunk(await File.ReadAllTextAsync(file), (int)parameters.ContextSize));
                }
            }

            vectorDb = new();
            int count = 1;
            foreach (string s in trainingData)
            {
                Console.WriteLine($"Embedding chunk {count}/{trainingData.Count}");
                float[] embeddings = embedder.GetEmbeddings(s);
                vectorDb.Add(s, embeddings);
                count++;
            }

            SqliteOutput sqliteOutput = new SqliteOutput(args.VectorDbPath);
            sqliteOutput.Serialize(vectorDb);
            
            Console.WriteLine($"Embedded data saved to {args.VectorDbPath}");
        }
        else if (File.Exists(args.VectorDbPath))
        {
            vectorDb = SqliteInput.ReadFromVectorDb(args.VectorDbPath, embedder.EmbeddingSize);
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
        string? prompt = "Ask the author a question";
        Console.WriteLine(prompt);
        while (prompt != "exit")
        {
            prompt = Console.ReadLine();
            if(string.IsNullOrEmpty(prompt)) continue;
            float[] embeddings = embedder.GetEmbeddings(prompt);
            string topText = TextFromEmbedding(vectorDb, embeddings, 3);
            prompt = $"Using the text passages below, please answer the user's question in the voice of the author:\n\n {topText} \n\n Question: {prompt}";
            
            await foreach (string text in session.ChatAsync(
                               prompt,
                               new InferenceParams { Temperature = 0.6f }))
            {
                Console.Write(text);
            }
        }
    }

    static IEnumerable<string> Chunk(string str, int chunkSize)
    {
        return Enumerable.Range(0, str.Length / chunkSize)
            .Select(i => str.Substring(i * chunkSize, chunkSize));
    }

    static string TextFromEmbedding(Dictionary<string, float[]> vectorDb, IReadOnlyList<float> promptEmbedding, int numTop)
    {
        Dictionary<float, string> topScored = new();
        foreach ((string text, float[] vector) in vectorDb)
        {
            float score = DotProduct(vector, promptEmbedding);
            topScored.Add(score, text);
        }

        ImmutableSortedDictionary<float, string> sorted = topScored.ToImmutableSortedDictionary(new DescendingComparer<float>());

        return string.Join("\n", sorted.Values.Take(numTop));
    }

    static float DotProduct(IReadOnlyCollection<float> vectorA, IReadOnlyList<float> vectorB)
    {
        if (vectorA.Count != vectorB.Count)
        {
            throw new ArgumentException("Vectors must be of the same dimension.");
        }

        return vectorA.Select((t, i) => t * vectorB[i]).Sum();
    }
    
    class DescendingComparer<T> : IComparer<T> where T : IComparable<T> {
        public int Compare(T x, T y) {
            return y.CompareTo(x);
        }
    }
}