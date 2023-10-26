using System.Data;
using System.Data.SQLite;

namespace gguf_RAG;

/// <summary>
/// Writes the result to an external database.
/// </summary>
public class SqliteOutput
{
    readonly string DbPath;

    public SqliteOutput(string dbPath)
    {
        DbPath = dbPath;
    }

    public void Serialize(Dictionary<string, float[]> vectorDb)
    {
        if (string.IsNullOrEmpty(DbPath)) return;

        if (File.Exists(DbPath))
        {
            File.Delete(DbPath);
        }

        string cs = $"URI=file:{DbPath}";

        using SQLiteConnection conn = new(cs);
        conn.Open();

        using (IDbCommand cmd = conn.CreateCommand())
        {
            cmd.CommandText =
                @"CREATE TABLE texts (
                          id INTEGER PRIMARY KEY ASC,
                          text TEXT NOT NULL
                      )";
            cmd.ExecuteNonQuery();

            cmd.CommandText =
                @"CREATE TABLE vectors (
                          id INTEGER PRIMARY KEY ASC,
                          vector NUMBER NOT NULL
                      )";
            cmd.ExecuteNonQuery();
        }

        InsertData(conn, vectorDb);
    }

    static void InsertData(
        IDbConnection conn,
        Dictionary<string, float[]> vectorDb)
    {
        using IDbCommand cmd = conn.CreateCommand();
        using IDbTransaction transaction = conn.BeginTransaction();
        foreach ((string text, float[] vector) in vectorDb)
        {
            cmd.CommandText =
                @"INSERT INTO texts (
                          text
                      ) VALUES (
                          @Text
                      )";

            IDbDataParameter textParameter =
                cmd.CreateParameter();
            textParameter.DbType = DbType.String;
            textParameter.ParameterName = "@Text";
            textParameter.Value = text;
            cmd.Parameters.Add(textParameter);

            cmd.ExecuteNonQuery();

            cmd.CommandText =
                @"INSERT INTO vectors (
                          vector
                      ) VALUES (
                          @Vector
                      )";

            foreach (float f in vector)
            {
                IDbDataParameter vectorParameter =
                    cmd.CreateParameter();
                vectorParameter.DbType = DbType.Single;
                vectorParameter.ParameterName = "@Vector";
                vectorParameter.Value = f;
                cmd.Parameters.Add(vectorParameter);

                cmd.ExecuteNonQuery();    
            }
        }

        transaction.Commit();
    }
}