<?php

namespace App\Controllers;

use CodeIgniter\Controller;
use CodeIgniter\HTTP\ResponseInterface;

class Project extends Controller
{
    private string $socketHost = '127.0.0.1';
    private int $socketPort = 16888;

    public function index(): string
    {
        return $this->renderProjectPage(['result' => null, 'error' => null, 'old' => []]);
    }

    public function predict(): string
    {
        $old = [
            'transcript_id' => (string) $this->request->getPost('transcript_id'),
            'utr5' => (string) $this->request->getPost('utr5'),
            'cds' => (string) $this->request->getPost('cds'),
            'utr3' => (string) $this->request->getPost('utr3'),
            'threshold' => (string) ($this->request->getPost('threshold') ?? '0.5'),
        ];

        try {
            $payload = [
                'action' => 'predict',
                'transcript_id' => $old['transcript_id'] ?: 'query',
                '5UTRseq' => $old['utr5'],
                'CDSseq' => $old['cds'],
                '3UTRseq' => $old['utr3'],
                'threshold' => (float) $old['threshold'],
            ];
            $response = $this->sendSocketRequest($payload);
            $result = $response['prediction'] ?? null;
            $error = null;
        } catch (\Throwable $exception) {
            $result = null;
            $error = $exception->getMessage();
        }

        return $this->renderProjectPage(['result' => $result, 'error' => $error, 'old' => $old]);
    }

    public function apiPredict(): ResponseInterface
    {
        try {
            $payload = $this->request->getJSON(true) ?? [];
            $payload['action'] = 'predict';
            return $this->response->setJSON($this->sendSocketRequest($payload));
        } catch (\Throwable $exception) {
            return $this->response
                ->setStatusCode(400)
                ->setJSON(['ok' => false, 'error' => $exception->getMessage()]);
        }
    }

    public function health(): ResponseInterface
    {
        try {
            return $this->response->setJSON($this->sendSocketRequest(['action' => 'health']));
        } catch (\Throwable $exception) {
            return $this->response
                ->setStatusCode(503)
                ->setJSON(['ok' => false, 'error' => $exception->getMessage()]);
        }
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

    private function renderProjectPage(array $data): string
    {
        helper(['form', 'url']);

        return view('layout/header')
            . view('project/index', $data)
            . view('layout/footer');
    }
}
