package com.example.animanpu.runtime

class OrtSessionFactory {
    fun buildQnnProviderOptions(config: OrtQnnConfig): Map<String, String> = config.providerOptions()
}
