hdir=$(pwd)
cd ..

mkdir datasets
mkdir models
mkdir datasets/external

cd datasets

kaggle competitions download -c llm-detect-ai-generated-text
unzip llm-detect-ai-generated-text.zip -d llm-detect-ai-generated-text
rm llm-detect-ai-generated-text.zip

kaggle datasets download -d nbroad/persaude-corpus-2
unzip persaude-corpus-2.zip -d ./
rm persaude-corpus-2.zip

kaggle datasets download -d conjuring92/ai-mix-v16
unzip ai-mix-v16.zip -d ./external/ai_mix_v16
rm ai-mix-v16.zip

kaggle datasets download -d conjuring92/ai-mix-v26
unzip ai-mix-v26.zip -d ./external/ai_mix_v26
rm ai-mix-v26.zip

kaggle datasets download -d conjuring92/ai-bin7-mix-v1
unzip ai-bin7-mix-v1.zip -d ./external/ai_mix_for_ranking
rm ai-bin7-mix-v1.zip

cd ../models
kaggle datasets download -d conjuring92/detect-ai-persuade-clm-ckpts
unzip detect-ai-persuade-clm-ckpts.zip -d ./
rm detect-ai-persuade-clm-ckpts.zip

cd $hdir
