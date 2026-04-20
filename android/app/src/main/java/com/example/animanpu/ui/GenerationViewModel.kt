package com.example.animanpu.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.animanpu.runtime.GenerationOrchestrator
import com.example.animanpu.runtime.GenerationRequest
import com.example.animanpu.runtime.GenerationResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class GenerationUiState(
    val prompt: String = "",
    val negativePrompt: String = "",
    val resolution: String = "1024x1024",
    val steps: String = "12",
    val cfg: String = "5.0",
    val isGenerating: Boolean = false,
    val lastResult: GenerationResult? = null,
    val error: String? = null,
)

class GenerationViewModel(
    private val orchestrator: GenerationOrchestrator,
) : ViewModel() {
    private val _state = MutableStateFlow(GenerationUiState())
    val state: StateFlow<GenerationUiState> = _state.asStateFlow()

    fun updatePrompt(value: String) {
        _state.value = _state.value.copy(prompt = value)
    }

    fun updateNegativePrompt(value: String) {
        _state.value = _state.value.copy(negativePrompt = value)
    }

    fun updateResolution(value: String) {
        _state.value = _state.value.copy(resolution = value)
    }

    fun updateSteps(value: String) {
        _state.value = _state.value.copy(steps = value)
    }

    fun updateCfg(value: String) {
        _state.value = _state.value.copy(cfg = value)
    }

    fun generate() {
        val parts = _state.value.resolution.split("x")
        val request = GenerationRequest(
            width = parts[0].toInt(),
            height = parts[1].toInt(),
            prompt = _state.value.prompt,
            negativePrompt = _state.value.negativePrompt,
            steps = _state.value.steps.toInt(),
            cfg = _state.value.cfg.toFloat(),
        )
        _state.value = _state.value.copy(isGenerating = true, error = null)
        viewModelScope.launch {
            runCatching { orchestrator.generate(request) }
                .onSuccess { result ->
                    _state.value = _state.value.copy(isGenerating = false, lastResult = result)
                }
                .onFailure { error ->
                    _state.value = _state.value.copy(isGenerating = false, error = error.message)
                }
        }
    }
}
