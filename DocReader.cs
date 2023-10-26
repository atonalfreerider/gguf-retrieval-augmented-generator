using PdfSharp.Pdf;
using PdfSharp.Pdf.IO;

namespace gguf_RAG;

public static class DocReader
{
    public static async Task<List<string>> ReadFolder(string folderPath, int chunkSize)
    {
        List<string> trainingData = new();
        foreach (string file in Directory.EnumerateFiles(folderPath))
        {
            Console.WriteLine($"reading {file}");
               
            if (file.EndsWith(".pdf"))
            {
                PdfDocument pdfDocument = PdfReader.Open(file);
                foreach (PdfPage page in pdfDocument.Pages)
                {
                    string agg = string.Join("", page.ExtractText());
                    trainingData.AddRange(Chunk(agg, chunkSize));
                }
            }
            else if (file.EndsWith(".txt"))
            {
                trainingData.AddRange(Chunk(await File.ReadAllTextAsync(file), chunkSize));
            }
        }

        return trainingData;
    }
    
    static IEnumerable<string> Chunk(string str, int chunkSize)
    {
        return Enumerable.Range(0, str.Length / chunkSize)
            .Select(i => str.Substring(i * chunkSize, chunkSize));
    }
}