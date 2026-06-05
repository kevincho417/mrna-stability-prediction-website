<?php

namespace App\Models;

use CodeIgniter\Model;

class PredictionModel extends Model
{
    private string $socketHost = '127.0.0.1';
    private int $socketPort = 16888;
    private string $databasePath;

    public function __construct()
    {
        // The project keeps history in a standalone SQLite file under writable/.
        $this->databasePath = WRITEPATH . 'prediction_history.sqlite';
        $this->initializeDatabase();
    }

    public function predict(array $payload): array
    {
        $response = $this->sendSocketRequest($this->normalizePredictionPayload($payload));
        $prediction = $response['prediction'] ?? null;

        if (! is_array($prediction)) {
            throw new \RuntimeException('Prediction socket response did not include a prediction object.');
        }

        $this->storeHistory($prediction);

        return $prediction;
    }

    public function health(): array
    {
        return $this->sendSocketRequest(['action' => 'health']);
    }

    public function getRecentHistory(int $limit = 10): array
    {
        $limit = max(1, min($limit, 100));
        $database = $this->connectDatabase();
        $result = $database->query(
            'SELECT id, created_at, transcript_id, predicted_label, predicted_probability, class_name,
                    threshold, mlp_probability, transformer_probability, utr5_length, cds_length,
                    utr3_length, total_length
             FROM prediction_history
             ORDER BY id DESC
             LIMIT ' . $limit
        );

        $history = [];
        while ($row = $result->fetchArray(SQLITE3_ASSOC)) {
            $history[] = $row;
        }

        $database->close();

        return $history;
    }

    private function normalizePredictionPayload(array $payload): array
    {
        return [
            'action' => 'predict',
            'transcript_id' => (string) ($payload['transcript_id'] ?? $payload['TranscriptID'] ?? 'query'),
            '5UTRseq' => (string) ($payload['5UTRseq'] ?? $payload['utr5'] ?? ''),
            'CDSseq' => (string) ($payload['CDSseq'] ?? $payload['cds'] ?? ''),
            '3UTRseq' => (string) ($payload['3UTRseq'] ?? $payload['utr3'] ?? ''),
            'threshold' => (float) ($payload['threshold'] ?? 0.5),
        ];
    }

    private function storeHistory(array $prediction): void
    {
        $lengths = $prediction['sequence_lengths'] ?? [];
        $database = $this->connectDatabase();
        $statement = $database->prepare(
            'INSERT INTO prediction_history (
                created_at, transcript_id, predicted_label, predicted_probability, class_name,
                threshold, mlp_probability, transformer_probability, utr5_length, cds_length,
                utr3_length, total_length
            ) VALUES (
                :created_at, :transcript_id, :predicted_label, :predicted_probability, :class_name,
                :threshold, :mlp_probability, :transformer_probability, :utr5_length, :cds_length,
                :utr3_length, :total_length
            )'
        );

        $statement->bindValue(':created_at', date('Y-m-d H:i:s'), SQLITE3_TEXT);
        $statement->bindValue(':transcript_id', (string) ($prediction['transcript_id'] ?? 'query'), SQLITE3_TEXT);
        $statement->bindValue(':predicted_label', (int) ($prediction['predicted_label'] ?? 0), SQLITE3_INTEGER);
        $statement->bindValue(':predicted_probability', (float) ($prediction['predicted_probability'] ?? 0), SQLITE3_FLOAT);
        $statement->bindValue(':class_name', (string) ($prediction['class_name'] ?? ''), SQLITE3_TEXT);
        $statement->bindValue(':threshold', (float) ($prediction['threshold'] ?? 0.5), SQLITE3_FLOAT);
        $statement->bindValue(':mlp_probability', (float) ($prediction['mlp_probability'] ?? 0), SQLITE3_FLOAT);
        $statement->bindValue(':transformer_probability', (float) ($prediction['transformer_probability'] ?? 0), SQLITE3_FLOAT);
        $statement->bindValue(':utr5_length', (int) ($lengths['5UTRseq'] ?? 0), SQLITE3_INTEGER);
        $statement->bindValue(':cds_length', (int) ($lengths['CDSseq'] ?? 0), SQLITE3_INTEGER);
        $statement->bindValue(':utr3_length', (int) ($lengths['3UTRseq'] ?? 0), SQLITE3_INTEGER);
        $statement->bindValue(':total_length', (int) ($lengths['total'] ?? 0), SQLITE3_INTEGER);

        $statement->execute();
        $database->close();
    }

    private function initializeDatabase(): void
    {
        if (! class_exists(\SQLite3::class)) {
            throw new \RuntimeException('PHP SQLite3 extension is not installed. Run: sudo apt install sqlite3 php-sqlite3');
        }

        $directory = dirname($this->databasePath);
        if (! is_dir($directory)) {
            mkdir($directory, 0775, true);
        }

        $database = $this->connectDatabase();
        $database->exec(
            'CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                transcript_id TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                predicted_probability REAL NOT NULL,
                class_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                mlp_probability REAL NOT NULL,
                transformer_probability REAL NOT NULL,
                utr5_length INTEGER NOT NULL,
                cds_length INTEGER NOT NULL,
                utr3_length INTEGER NOT NULL,
                total_length INTEGER NOT NULL
            )'
        );
        $database->exec(
            'CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at
             ON prediction_history(created_at DESC)'
        );
        $database->close();
    }

    private function connectDatabase(): \SQLite3
    {
        $database = new \SQLite3($this->databasePath);
        $database->enableExceptions(true);
        $database->busyTimeout(5000);

        return $database;
    }

    private function sendSocketRequest(array $payload): array
    {
        $connection = @fsockopen($this->socketHost, $this->socketPort, $errno, $errstr, 60);
        if ($connection === false) {
            throw new \RuntimeException("Unable to connect to prediction socket {$this->socketHost}:{$this->socketPort}: {$errstr}");
        }

        stream_set_timeout($connection, 60);
        fwrite($connection, json_encode($payload) . "\n");
        $line = fgets($connection);
        fclose($connection);

        if ($line === false || trim($line) === '') {
            throw new \RuntimeException('Prediction socket returned an empty response.');
        }

        $response = json_decode($line, true);
        if (! is_array($response)) {
            throw new \RuntimeException('Prediction socket returned invalid JSON.');
        }
        if (($response['ok'] ?? false) !== true) {
            throw new \RuntimeException($response['error'] ?? 'Prediction socket returned an error.');
        }

        return $response;
    }
}
