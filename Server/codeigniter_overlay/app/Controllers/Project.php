<?php

namespace App\Controllers;

use App\Models\PredictionModel;
use CodeIgniter\Controller;
use CodeIgniter\HTTP\ResponseInterface;

class Project extends Controller
{
    public function index(): string
    {
        return $this->renderProjectPage('project/index', [
            'result' => null,
            'error' => null,
            'old' => [],
        ]);
    }

    public function predict(): string
    {
        $model = new PredictionModel();
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
            $result = $model->predict($payload);
            $error = null;
        } catch (\Throwable $exception) {
            $result = null;
            $error = $exception->getMessage();
        }

        return $this->renderProjectPage('project/index', [
            'result' => $result,
            'error' => $error,
            'old' => $old,
        ]);
    }

    public function history(): string
    {
        $model = new PredictionModel();

        return $this->renderProjectPage('project/history', [
            'history' => $model->getRecentHistory(50),
        ]);
    }

    public function healthPage(): string
    {
        $health = null;
        $healthRows = [];
        $error = null;

        try {
            $model = new PredictionModel();
            $health = $model->health();
            $healthRows = $this->flattenTableRows($health);
        } catch (\Throwable $exception) {
            $error = $exception->getMessage();
        }

        return $this->renderProjectPage('project/health', [
            'health' => $health,
            'healthRows' => $healthRows,
            'error' => $error,
        ]);
    }

    public function apiPredict(): ResponseInterface
    {
        try {
            $model = new PredictionModel();
            $payload = $this->request->getJSON(true) ?? [];

            return $this->response->setJSON([
                'ok' => true,
                'prediction' => $model->predict($payload),
            ]);
        } catch (\Throwable $exception) {
            return $this->response
                ->setStatusCode(400)
                ->setJSON(['ok' => false, 'error' => $exception->getMessage()]);
        }
    }

    public function health(): ResponseInterface
    {
        try {
            $model = new PredictionModel();

            return $this->response->setJSON($model->health());
        } catch (\Throwable $exception) {
            return $this->response
                ->setStatusCode(503)
                ->setJSON(['ok' => false, 'error' => $exception->getMessage()]);
        }
    }

    private function renderProjectPage(string $view, array $data): string
    {
        helper(['form', 'url']);

        return view('layout/header')
            . view($view, $data)
            . view('layout/footer');
    }

    private function flattenTableRows(array $data, string $prefix = ''): array
    {
        $rows = [];
        foreach ($data as $key => $value) {
            $label = $prefix === '' ? (string) $key : $prefix . '.' . $key;

            if (is_array($value)) {
                $rows = array_merge($rows, $this->flattenTableRows($value, $label));
                continue;
            }

            $rows[] = [
                'label' => $label,
                'value' => $this->formatHealthValue($value),
            ];
        }

        return $rows;
    }

    private function formatHealthValue(mixed $value): string
    {
        if (is_bool($value)) {
            return $value ? 'true' : 'false';
        }
        if ($value === null) {
            return 'null';
        }

        return (string) $value;
    }
}
