using System.Collections.Immutable;
using LLama;

namespace gguf_RAG;

public class VectorDb
{
    readonly LLamaEmbedder Embedder;
    public Dictionary<string, float[]> VectorsByChunk = new();
    public VectorDb(LLamaEmbedder embedder)
    {
        Embedder = embedder;
    }
    
    public void Train(List<string> trainingData)
    {
        int count = 1;
        foreach (string s in trainingData)
        {
            Console.WriteLine($"Embedding chunk {count}/{trainingData.Count}");
            float[] embeddings = Embedder.GetEmbeddings(s);
            VectorsByChunk.Add(s, embeddings);
            count++;
        }
    }
    
    public string TextFromEmbedding(IReadOnlyList<float> promptEmbedding, int numTop)
    {
        Dictionary<float, string> topScored = new();
        foreach ((string text, float[] vector) in VectorsByChunk)
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