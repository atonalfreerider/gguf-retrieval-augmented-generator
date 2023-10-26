using System.Data;
using System.Data.SQLite;

namespace gguf_RAG;

public static class SqliteInput
{
    public static Dictionary<string, float[]> ReadFromVectorDb(string DbPath, int embeddingSize)
    {
        string cs = $"URI=file:{DbPath}";

        using SQLiteConnection conn = new(cs);
        conn.Open();

        List<string> columnNames = new List<string>
        {
            "id", "text"
        };

        using IDbCommand cmd = conn.CreateCommand();

        cmd.CommandText = CommandString(columnNames, "texts");
        List<string> texts = new();
        using (IDataReader reader = cmd.ExecuteReader())
        {
            Dictionary<string, int> indexes = ColumnIndexes(reader, columnNames);

            while (reader.Read())
            {
                string text = reader.GetString(indexes["text"]);
                texts.Add(text);
            }

            columnNames = new List<string>
            {
                "id", "vector"
            };
        }

        cmd.CommandText = CommandString(columnNames, "vectors");
        List<float> vectors = new();
        using (IDataReader reader = cmd.ExecuteReader())
        {
            Dictionary<string, int> indexes = ColumnIndexes(reader, columnNames);

            while (reader.Read())
            {
                float vector = reader.GetFloat(indexes["vector"]);
                vectors.Add(vector);
            }
        }

        Dictionary<string, float[]> vectorDb = new();
        int count = 0;
        foreach (string text in texts)
        {
            float[] vectorValues = new float[embeddingSize];
            for (int i = count * embeddingSize; i < (count + 1) * embeddingSize; i++)
            {
                vectorValues[i - count * embeddingSize] = vectors[i];
            }

            vectorDb.Add(text, vectorValues);
            count++;
        }

        return vectorDb;
    }

    static string CommandString(IEnumerable<string> columnNames, string tableName)
    {
        string cmd = columnNames.Aggregate(
            "SELECT ",
            (current, columnName) => current + $"{columnName}, ");

        // remove last comma
        cmd = cmd[..^2] + " ";
        cmd += $"FROM {tableName}";

        return cmd;
    }

    static Dictionary<string, int> ColumnIndexes(IDataRecord reader, IEnumerable<string> columnNames)
    {
        return columnNames
            .ToDictionary(
                columnName => columnName,
                reader.GetOrdinal);
    }
}