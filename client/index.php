<?php

$defaultBaseUrl = 'http://127.0.0.1:8000';
$result = null;
$error = null;
$selectedEndpoint = $_POST['endpoint'] ?? '/inference';
$baseUrl = rtrim($_POST['base_url'] ?? $defaultBaseUrl, '/');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!extension_loaded('curl')) {
        $error = 'The PHP cURL extension is required.';
    } elseif (!isset($_FILES['json_file']) || $_FILES['json_file']['error'] !== UPLOAD_ERR_OK) {
        $error = 'Please upload a valid JSON file.';
    } else {
        $uploadedPath = $_FILES['json_file']['tmp_name'];
        $rawJson = file_get_contents($uploadedPath);

        if ($rawJson === false) {
            $error = 'Could not read the uploaded file.';
        } else {
            json_decode($rawJson, true);
            if (json_last_error() !== JSON_ERROR_NONE) {
                $error = 'Uploaded file is not valid JSON: ' . json_last_error_msg();
            } else {
                $url = $baseUrl . $selectedEndpoint;
                $ch = curl_init($url);

                curl_setopt_array($ch, [
                    CURLOPT_POST => true,
                    CURLOPT_RETURNTRANSFER => true,
                    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
                    CURLOPT_POSTFIELDS => $rawJson,
                    CURLOPT_TIMEOUT => 30,
                ]);

                $responseBody = curl_exec($ch);
                $curlError = curl_error($ch);
                $statusCode = (int) curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
                curl_close($ch);

                if ($responseBody === false) {
                    $error = 'Request failed: ' . $curlError;
                } else {
                    $result = [
                        'url' => $url,
                        'status' => $statusCode,
                        'request_body' => $rawJson,
                        'response_body' => $responseBody,
                    ];
                }
            }
        }
    }
}

function h(string $value): string
{
    return htmlspecialchars($value, ENT_QUOTES, 'UTF-8');
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inference API Test Client</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 960px;
            margin: 40px auto;
            padding: 0 16px;
            line-height: 1.5;
            color: #1f2937;
        }

        h1, h2 {
            margin-bottom: 0.5rem;
        }

        form {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            background: #f9fafb;
        }

        label {
            display: block;
            font-weight: 600;
            margin-top: 12px;
            margin-bottom: 6px;
        }

        input[type="text"],
        select,
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            box-sizing: border-box;
        }

        button {
            margin-top: 16px;
            padding: 10px 16px;
            border: 0;
            border-radius: 6px;
            background: #111827;
            color: #ffffff;
            cursor: pointer;
        }

        .note,
        .error,
        .result {
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .note {
            background: #eef2ff;
            border: 1px solid #c7d2fe;
        }

        .error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }

        .result {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
        }

        pre {
            white-space: pre-wrap;
            word-break: break-word;
            background: #111827;
            color: #f9fafb;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
        }

        code {
            background: #e5e7eb;
            padding: 2px 4px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>Inference API Test Client</h1>

    <div class="note">
        <p>Run the FastAPI app first, for example at <code>http://127.0.0.1:8000</code>.</p>
        <p>Use <code>single_item.json</code> with <code>/inference</code> and <code>batch_items.json</code> with <code>/batch_inference</code>.</p>
    </div>

    <?php if ($error !== null): ?>
        <div class="error">
            <strong>Error:</strong> <?= h($error) ?>
        </div>
    <?php endif; ?>

    <?php if ($result !== null): ?>
        <div class="result">
            <h2>Response</h2>
            <p><strong>URL:</strong> <?= h($result['url']) ?></p>
            <p><strong>Status:</strong> <?= h((string) $result['status']) ?></p>
            <h2>Request JSON</h2>
            <pre><?= h($result['request_body']) ?></pre>
            <h2>Response Body</h2>
            <pre><?= h($result['response_body']) ?></pre>
        </div>
    <?php endif; ?>

    <form method="post" enctype="multipart/form-data">
        <label for="base_url">Base URL</label>
        <input id="base_url" name="base_url" type="text" value="<?= h($baseUrl) ?>" required>

        <label for="endpoint">Endpoint</label>
        <select id="endpoint" name="endpoint">
            <option value="/inference" <?= $selectedEndpoint === '/inference' ? 'selected' : '' ?>>/inference</option>
            <option value="/batch_inference" <?= $selectedEndpoint === '/batch_inference' ? 'selected' : '' ?>>/batch_inference</option>
        </select>

        <label for="json_file">Upload JSON File</label>
        <input id="json_file" name="json_file" type="file" accept=".json,application/json" required>

        <button type="submit">Send POST Request</button>
    </form>
</body>
</html>
