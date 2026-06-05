<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>mRNA Stability Prediction</title>
    <link rel="stylesheet" href="<?= base_url('assets/styles.css') ?>">
</head>
<body class="min-h-screen bg-mist-50 text-zinc-900 font-sans antialiased">
    <header class="border-b border-mist-200 bg-white">
        <div class="mx-auto flex min-h-20 w-[min(1180px,calc(100%_-_32px))] items-center justify-between gap-6 max-md:block max-md:py-4">
            <div>
                <h1 class="text-2xl font-bold">mRNA Stability Prediction</h1>
                <p class="mt-1.5 text-sm text-zinc-500">DL & Big Data 2026 Project</p>
            </div>
            <nav class="flex gap-2 max-md:mt-4" aria-label="Server links">
                <a class="rounded-md border border-mist-200 px-3 py-2 text-sm font-bold text-cyan-800 hover:bg-mist-50" href="<?= site_url('2026Project') ?>">Predict</a>
                <a class="rounded-md border border-mist-200 px-3 py-2 text-sm font-bold text-cyan-800 hover:bg-mist-50" href="<?= site_url('2026Project/history') ?>">History</a>
                <a class="rounded-md border border-mist-200 px-3 py-2 text-sm font-bold text-cyan-800 hover:bg-mist-50" href="<?= site_url('2026Project/health') ?>">Health</a>
            </nav>
        </div>
    </header>

    <main class="mx-auto my-6 w-[min(1180px,calc(100%_-_32px))]">
