using System;

namespace BerdyGPT
{
    public class TrainingExample
    {
        public int[] ModelInput { get; private set; }
        public int[] Targets { get; private set; }

        private const int MaskingIndex = -100;

        public TrainingExample(int[] inputTokens, int[] outputTokens, int eosToken)
        {
            int[] fullSequence = new int[inputTokens.Length + outputTokens.Length + eosToken];

            int index = 0;

            for (int i = 0; i < inputTokens.Length; i++)
            {
                fullSequence[index] = inputTokens[i];
                index++;
            }

            for (int i = 0; i < outputTokens.Length; i++)
            {
                fullSequence[index] = outputTokens[i];
                index++;
            }

            fullSequence[index] = eosToken;

            ModelInput = new int[fullSequence.Length - 1];
            Targets = new int[fullSequence.Length - 1];

            int firstResponseTargetIndex = fullSequence.Length - 1;

            for (int i = 0; i < fullSequence.Length - 1; i++)
            {
                ModelInput[i] = fullSequence[i];
                Targets[i] = (i >= firstResponseTargetIndex - 1 ? fullSequence[i + 1] : MaskingIndex);
            }
        }
    }
}