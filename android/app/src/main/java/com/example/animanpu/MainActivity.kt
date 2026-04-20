package com.example.animanpu

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.example.animanpu.runtime.GenerationEngine
import com.example.animanpu.runtime.GenerationOrchestrator
import com.example.animanpu.runtime.GenerationRequest
import com.example.animanpu.runtime.GenerationResult
import com.example.animanpu.ui.GenerationScreen
import com.example.animanpu.ui.GenerationViewModel

class MainActivity : ComponentActivity() {
    private val viewModel by lazy {
        val fakeEngine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult {
                return GenerationResult(
                    imagePath = "/sdcard/Download/anima-output.png",
                    totalDurationMs = 1200,
                    denoiseDurationMs = 1000,
                    profilingPath = "/sdcard/Download/profile.csv",
                    qnnActive = false,
                )
            }
        }
        GenerationViewModel(GenerationOrchestrator(fakeEngine))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val state by viewModel.state.collectAsState()
            GenerationScreen(
                state = state,
                onPromptChange = viewModel::updatePrompt,
                onNegativePromptChange = viewModel::updateNegativePrompt,
                onStepsChange = viewModel::updateSteps,
                onCfgChange = viewModel::updateCfg,
                onGenerate = viewModel::generate,
            )
        }
    }
}
