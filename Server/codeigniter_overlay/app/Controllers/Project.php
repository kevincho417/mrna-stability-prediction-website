<?php

namespace App\Controllers;

use App\Models\PredictionModel;
use CodeIgniter\Controller;
use CodeIgniter\HTTP\ResponseInterface;

class Project extends Controller
{
    public function index(): string
    {
        $model = new PredictionModel();

        return $this->renderProjectPage([
            'result' => null,
            'error' => null,
            'old' => [],
            'history' => $model->getRecentHistory(),
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

        return $this->renderProjectPage([
            'result' => $result,
            'error' => $error,
            'old' => $old,
            'history' => $model->getRecentHistory(),
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

    private function renderProjectPage(array $data): string
    {
        helper(['form', 'url']);

        return view('layout/header')
            . view('project/index', $data)
            . view('layout/footer');
    }
}
