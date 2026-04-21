package com.example.animanpu

import android.os.Bundle
import java.io.File
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.example.animanpu.runtime.GenerationOrchestrator
import com.example.animanpu.runtime.ArtifactManager
import com.example.animanpu.runtime.OrtQnnDenoiserEngine
import com.example.animanpu.runtime.ReflectionOrtSessionFactory
import com.example.animanpu.ui.GenerationScreen
import com.example.animanpu.ui.GenerationViewModel

class MainActivity : ComponentActivity() {
    private val viewModel by lazy {
        val artifactManager = ArtifactManager(filesDir)
        val backendPath = File(applicationInfo.nativeLibraryDir, "libQnnHtp.so").path
        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifactManager,
            sessionFactory = ReflectionOrtSessionFactory(),
            backendPath = backendPath,
        )
        GenerationViewModel(GenerationOrchestrator(engine))
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
