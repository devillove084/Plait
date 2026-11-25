use crate::OperatorError;

use super::{
    DecoderStack, DecoderStackKv, LinearProjection, decode, finite_half, project, rms_norm,
};

/// Token embedding, final RMS normalization and vocabulary projection for a
/// decoder stack. Embeddings are row-major binary16 words in little-endian order.
pub struct TokenHead<P> {
    embeddings: Vec<u8>,
    norm: Vec<f32>,
    projection: P,
    epsilon: f32,
}

impl<P: LinearProjection> TokenHead<P> {
    pub fn new(
        embeddings: Vec<u8>,
        norm: &[u16],
        projection: P,
        epsilon: f32,
    ) -> Result<Self, OperatorError> {
        let (hidden, vocab) = projection.shape();
        let expected = (hidden as usize)
            .checked_mul(vocab as usize)
            .and_then(|n| n.checked_mul(2));
        if hidden == 0
            || vocab == 0
            || Some(embeddings.len()) != expected
            || norm.len() != hidden as usize
            || !finite_half(norm)
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || embeddings
                .chunks_exact(2)
                .any(|word| !decode(u16::from_le_bytes([word[0], word[1]])).is_finite())
        {
            return Err(OperatorError::Shape(
                "invalid token embeddings or vocabulary projection",
            ));
        }
        Ok(Self {
            embeddings,
            norm: norm.iter().map(|&bits| decode(bits)).collect(),
            projection,
            epsilon,
        })
    }

    /// Greedy decoding. The last selected token is returned before its KV is
    /// appended, so it can be the first input after an executor handoff.
    pub fn generate(
        &self,
        stack: &DecoderStack<impl LinearProjection>,
        cache: &mut DecoderStackKv,
        input: &[u32],
        max_new_tokens: u32,
        stop: &[u32],
    ) -> Result<Vec<u32>, OperatorError> {
        let (hidden, vocab) = self.projection.shape();
        if input.is_empty()
            || max_new_tokens == 0
            || input.iter().chain(stop).any(|&token| token >= vocab)
            || stack.blocks[0].config.hidden_size != hidden
        {
            return Err(OperatorError::Shape(
                "invalid token IDs, generation limit or decoder width",
            ));
        }
        let before = cache.tokens;
        let result = self.generate_steps(stack, cache, input, max_new_tokens, stop);
        if result.is_err() {
            cache.truncate(before);
        }
        result
    }

    /// Convert token IDs to row-major embedding activations.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, OperatorError> {
        let (width, vocab) = self.projection.shape();
        if tokens.iter().any(|&token| token >= vocab) {
            return Err(OperatorError::Shape(
                "invalid token ID for embedding lookup",
            ));
        }
        let row_bytes = width as usize * 2;
        Ok(tokens
            .iter()
            .flat_map(|&token| {
                let start = token as usize * row_bytes;
                self.embeddings[start..start + row_bytes]
                    .chunks_exact(2)
                    .map(|word| decode(u16::from_le_bytes([word[0], word[1]])))
            })
            .collect())
    }

    /// Return the greedy selection for the last hidden row.
    pub fn next_token(&self, hidden: &[f32]) -> Result<u32, OperatorError> {
        let width = self.norm.len();
        if hidden.is_empty() || hidden.len() % width != 0 {
            return Err(OperatorError::Shape(
                "token head requires complete hidden rows",
            ));
        }
        let normalized = rms_norm(&hidden[hidden.len() - width..], &self.norm, self.epsilon)?;
        let logits = project(&self.projection, &normalized, 1)?;
        let mut selected = 0;
        for (id, score) in logits.iter().enumerate() {
            if *score > logits[selected] {
                selected = id;
            }
        }
        Ok(selected as u32)
    }

    fn generate_steps(
        &self,
        stack: &DecoderStack<impl LinearProjection>,
        cache: &mut DecoderStackKv,
        input: &[u32],
        max_new_tokens: u32,
        stop: &[u32],
    ) -> Result<Vec<u32>, OperatorError> {
        let mut output = Vec::new();
        for _ in 0..max_new_tokens {
            let tokens = output.last().map_or(input, std::slice::from_ref);
            let embedded = self.embed(tokens)?;
            let hidden = stack.forward(&embedded, cache)?;
            let token = self.next_token(&hidden)?;
            output.push(token);
            if stop.contains(&token) {
                break;
            }
        }
        Ok(output)
    }
}
