<?php
$health = $health ?? null;
$healthRows = $healthRows ?? [];
$error = $error ?? null;
?>

<section class="rounded-lg border border-mist-200 bg-white p-5 shadow-sm shadow-mist-950/5">
    <div class="flex flex-wrap items-end justify-between gap-3">
        <div>
            <h2 class="text-base font-bold">Health Status</h2>
            <p class="mt-1 text-sm text-zinc-500">Server, socket, and model metadata.</p>
        </div>
        <?php if ($error): ?>
            <span class="rounded-md border border-red-200 bg-red-50 px-3 py-1.5 text-sm font-bold text-red-800">Unavailable</span>
        <?php else: ?>
            <span class="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-sm font-bold text-emerald-800">Online</span>
        <?php endif; ?>
    </div>

    <?php if ($error): ?>
        <div class="mt-4 rounded-lg border border-red-200 bg-red-50 p-4 text-red-800">
            <strong>Health check failed</strong>
            <p class="mt-2 text-sm text-red-700"><?= esc($error) ?></p>
        </div>
    <?php elseif ($healthRows === []): ?>
        <div class="mt-4 rounded-md border border-dashed border-mist-300 bg-mist-50 p-4 text-sm text-zinc-500">
            No health metadata returned.
        </div>
    <?php else: ?>
        <div class="mt-4 overflow-x-auto">
            <table class="w-full min-w-[760px] border-collapse text-sm">
                <thead>
                    <tr class="border-b border-mist-200 text-left text-xs uppercase text-zinc-500">
                        <th class="py-2 pr-4 font-bold">Field</th>
                        <th class="py-2 font-bold">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="border-b border-mist-200">
                        <td class="py-3 pr-4 font-bold">http_server</td>
                        <td class="py-3 font-mono text-zinc-700">Apache2 + CodeIgniter 4 / PHP</td>
                    </tr>
                    <tr class="border-b border-mist-200">
                        <td class="py-3 pr-4 font-bold">socket_endpoint</td>
                        <td class="py-3 font-mono text-zinc-700">127.0.0.1:16888</td>
                    </tr>
                    <?php foreach ($healthRows as $row): ?>
                        <tr class="border-b border-mist-200 last:border-b-0">
                            <td class="py-3 pr-4 font-bold"><?= esc($row['label']) ?></td>
                            <td class="py-3 font-mono text-zinc-700"><?= esc($row['value']) ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    <?php endif; ?>
</section>
