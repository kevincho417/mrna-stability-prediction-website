<?php
$history = $history ?? [];
?>

<section class="rounded-lg border border-mist-200 bg-white p-5 shadow-sm shadow-mist-950/5">
    <div class="flex flex-wrap items-end justify-between gap-3">
        <div>
            <h2 class="text-base font-bold">Prediction History</h2>
            <p class="mt-1 text-sm text-zinc-500">Saved SQL records from recent inference requests.</p>
        </div>
        <span class="rounded-md border border-mist-200 px-3 py-1.5 text-sm font-bold text-cyan-800">Latest 50</span>
    </div>

    <?php if ($history === []): ?>
        <div class="mt-4 rounded-md border border-dashed border-mist-300 bg-mist-50 p-4 text-sm text-zinc-500">
            No prediction history yet.
        </div>
    <?php else: ?>
        <div class="mt-4 overflow-x-auto">
            <table class="w-full min-w-[940px] border-collapse text-sm">
                <thead>
                    <tr class="border-b border-mist-200 text-left text-xs uppercase text-zinc-500">
                        <th class="py-2 pr-4 font-bold">Time</th>
                        <th class="py-2 pr-4 font-bold">Transcript ID</th>
                        <th class="py-2 pr-4 font-bold">Label</th>
                        <th class="py-2 pr-4 text-right font-bold">Ensemble</th>
                        <th class="py-2 pr-4 text-right font-bold">MLP</th>
                        <th class="py-2 pr-4 text-right font-bold">Transformer</th>
                        <th class="py-2 pr-4 text-right font-bold">Threshold</th>
                        <th class="py-2 text-right font-bold">Length</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($history as $item): ?>
                        <?php $isStable = (int) $item['predicted_label'] === 1; ?>
                        <tr class="border-b border-mist-200 last:border-b-0">
                            <td class="py-3 pr-4 whitespace-nowrap text-zinc-500"><?= esc($item['created_at']) ?></td>
                            <td class="py-3 pr-4 font-bold"><?= esc($item['transcript_id']) ?></td>
                            <td class="py-3 pr-4">
                                <span class="inline-flex rounded-md border px-2 py-1 text-xs font-bold <?= $isStable ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-red-200 bg-red-50 text-red-800' ?>">
                                    <?= esc((string) $item['predicted_label']) ?>
                                </span>
                                <span class="ml-2 text-zinc-500"><?= esc($item['class_name']) ?></span>
                            </td>
                            <td class="py-3 pr-4 text-right font-bold"><?= number_format((float) $item['predicted_probability'], 4) ?></td>
                            <td class="py-3 pr-4 text-right"><?= number_format((float) $item['mlp_probability'], 4) ?></td>
                            <td class="py-3 pr-4 text-right"><?= number_format((float) $item['transformer_probability'], 4) ?></td>
                            <td class="py-3 pr-4 text-right"><?= number_format((float) $item['threshold'], 2) ?></td>
                            <td class="py-3 text-right font-bold"><?= esc((string) $item['total_length']) ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    <?php endif; ?>
</section>
