package com.example.animanpu.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun GenerationScreen(
    state: GenerationUiState,
    onPromptChange: (String) -> Unit,
    onNegativePromptChange: (String) -> Unit,
    onStepsChange: (String) -> Unit,
    onCfgChange: (String) -> Unit,
    onGenerate: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = state.prompt,
            onValueChange = onPromptChange,
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.negativePrompt,
            onValueChange = onNegativePromptChange,
            label = { Text("Negative prompt") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.steps,
            onValueChange = onStepsChange,
            label = { Text("Steps") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.cfg,
            onValueChange = onCfgChange,
            label = { Text("CFG") },
            modifier = Modifier.fillMaxWidth(),
        )
        Button(onClick = onGenerate, enabled = !state.isGenerating) {
            Text(if (state.isGenerating) "Generating…" else "Generate")
        }
        state.lastResult?.let {
            Text("Last image: ${it.imagePath}")
            Text("Total: ${it.totalDurationMs} ms")
            Text("Denoise: ${it.denoiseDurationMs} ms")
            Text("QNN active: ${it.qnnActive}")
        }
        state.error?.let { Text("Error: $it") }
    }
}
