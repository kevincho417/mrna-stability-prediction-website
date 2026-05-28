<?php
$old = $old ?? [];
$result = $result ?? null;
$error = $error ?? null;

$transcriptId = esc($old['transcript_id'] ?? ($result['transcript_id'] ?? 'query'));
$threshold = esc((string) ($old['threshold'] ?? ($result['threshold'] ?? '0.5')));
$utr5 = esc($old['utr5'] ?? '');
$cds = esc($old['cds'] ?? '');
$utr3 = esc($old['utr3'] ?? '');
?>

<section class="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
    <form class="rounded-lg border border-mist-200 bg-white p-5 shadow-sm shadow-mist-950/5" method="post" action="<?= site_url('2026Project/predict') ?>">
        <?= csrf_field() ?>
        <div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_140px]">
            <label class="grid gap-2 text-sm font-bold">
                <span>Transcript ID</span>
                <input class="rounded-md border border-mist-200 bg-white px-3 py-2.5 text-zinc-900 outline-none focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20" name="transcript_id" value="<?= $transcriptId ?>">
            </label>
            <label class="grid gap-2 text-sm font-bold">
                <span>Threshold</span>
                <input class="rounded-md border border-mist-200 bg-white px-3 py-2.5 text-zinc-900 outline-none focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20" name="threshold" type="number" min="0" max="1" step="0.01" value="<?= $threshold ?>">
            </label>
        </div>

        <label class="mt-4 grid gap-2 text-sm font-bold">
            <span>5'UTR sequence</span>
            <textarea class="min-h-28 resize-y rounded-md border border-mist-200 bg-white px-3 py-2.5 font-mono text-sm leading-6 text-zinc-900 outline-none scrollbar-thin scrollbar-thumb-mist-300 scrollbar-track-transparent focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20" name="utr5" rows="5" spellcheck="false" placeholder="Optional"><?= $utr5 ?></textarea>
        </label>

        <label class="mt-4 grid gap-2 text-sm font-bold">
            <span>CDS sequence</span>
            <textarea class="min-h-40 resize-y rounded-md border border-mist-200 bg-white px-3 py-2.5 font-mono text-sm leading-6 text-zinc-900 outline-none scrollbar-thin scrollbar-thumb-mist-300 scrollbar-track-transparent focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20" name="cds" rows="8" spellcheck="false" required placeholder="Required, RNA sequence with A/U/C/G"><?= $cds ?></textarea>
        </label>

        <label class="mt-4 grid gap-2 text-sm font-bold">
            <span>3'UTR sequence</span>
            <textarea class="min-h-28 resize-y rounded-md border border-mist-200 bg-white px-3 py-2.5 font-mono text-sm leading-6 text-zinc-900 outline-none scrollbar-thin scrollbar-thumb-mist-300 scrollbar-track-transparent focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20" name="utr3" rows="5" spellcheck="false" placeholder="Optional"><?= $utr3 ?></textarea>
        </label>

        <div class="mt-4 flex justify-end">
            <button class="rounded-md bg-cyan-800 px-4 py-2.5 text-sm font-bold text-white hover:bg-cyan-900 focus:outline-none focus:ring-2 focus:ring-cyan-700/30" type="submit">Predict Stability</button>
        </div>
    </form>

    <aside class="rounded-lg border border-mist-200 bg-white p-5 shadow-sm shadow-mist-950/5">
        <?php if ($error): ?>
            <div class="rounded-lg border border-red-200 bg-red-50 p-4 text-red-800">
                <strong>Prediction failed</strong>
                <p class="mt-2 text-sm text-red-700"><?= esc($error) ?></p>
            </div>
        <?php elseif ($result): ?>
            <?php $isStable = (int) $result['predicted_label'] === 1; ?>
            <div class="rounded-lg border p-4 <?= $isStable ? 'border-emerald-200 bg-emerald-50' : 'border-red-200 bg-red-50' ?>">
                <span class="block text-sm text-zinc-500">Predicted label <?= esc((string) $result['predicted_label']) ?></span>
                <strong class="mt-1 block text-lg <?= $isStable ? 'text-emerald-800' : 'text-red-800' ?>"><?= esc($result['class_name']) ?></strong>
            </div>

            <dl class="mt-4 grid gap-2">
                <div class="flex justify-between gap-4 border-b border-mist-200 py-2.5">
                    <dt class="text-zinc-500">Ensemble probability</dt>
                    <dd class="font-bold"><?= number_format((float) $result['predicted_probability'], 4) ?></dd>
                </div>
                <div class="flex justify-between gap-4 border-b border-mist-200 py-2.5">
                    <dt class="text-zinc-500">MLP probability</dt>
                    <dd class="font-bold"><?= number_format((float) $result['mlp_probability'], 4) ?></dd>
                </div>
                <div class="flex justify-between gap-4 border-b border-mist-200 py-2.5">
                    <dt class="text-zinc-500">Transformer probability</dt>
                    <dd class="font-bold"><?= number_format((float) $result['transformer_probability'], 4) ?></dd>
                </div>
                <div class="flex justify-between gap-4 border-b border-mist-200 py-2.5">
                    <dt class="text-zinc-500">Threshold</dt>
                    <dd class="font-bold"><?= number_format((float) $result['threshold'], 2) ?></dd>
                </div>
            </dl>

            <div class="mt-5">
                <h2 class="mb-2 text-base font-bold">Sequence Lengths</h2>
                <table class="w-full border-collapse text-sm">
                    <tbody>
                        <tr class="border-b border-mist-200"><th class="py-2 text-left font-medium text-zinc-500">5'UTR</th><td class="py-2 text-right font-bold"><?= esc((string) $result['sequence_lengths']['5UTRseq']) ?></td></tr>
                        <tr class="border-b border-mist-200"><th class="py-2 text-left font-medium text-zinc-500">CDS</th><td class="py-2 text-right font-bold"><?= esc((string) $result['sequence_lengths']['CDSseq']) ?></td></tr>
                        <tr class="border-b border-mist-200"><th class="py-2 text-left font-medium text-zinc-500">3'UTR</th><td class="py-2 text-right font-bold"><?= esc((string) $result['sequence_lengths']['3UTRseq']) ?></td></tr>
                        <tr class="border-b border-mist-200"><th class="py-2 text-left font-medium text-zinc-500">Total</th><td class="py-2 text-right font-bold"><?= esc((string) $result['sequence_lengths']['total']) ?></td></tr>
                    </tbody>
                </table>
            </div>
        <?php else: ?>
            <div>
                <h2 class="text-base font-bold">Ready</h2>
                <p class="mt-2 text-sm text-zinc-500">Enter RNA sequences and run prediction.</p>
            </div>
        <?php endif; ?>
    </aside>
</section>
