from os import path
from libraries.log import logger
from libraries.strategies import *

import argparse

def describe(path2vectorizer, path2checkpoint, path2image, path2vocabulary, beam_width, path2ranker):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    logger.debug('env variables loading')
    logger.debug('features, vocab and token_ids loading')
    
    if path.isfile(path2checkpoint):
        logger.debug('model(snapshot) will be loaded')
        net = th.load(path2checkpoint)
        net.to(device)
        net.eval()

    vocab = deserialize(path2vocabulary)
    logger.debug(f'vocab was loaded | len => {len(vocab)}')
    
    logger.debug(f'load features extractor')

    vectorizer = load_vectorizer(path2vectorizer)
    vectorizer.eval()
    vectorizer.to(device)

    logger.debug('load ranker clip VIT model')
    ranker, processor = load_ranker(path2ranker, device)

    logger.debug('features extraction by resnet152')

    cv_image = read_image(path2image)
    th_image = cv2th(cv_image)
    th_image = prepare_image(th_image)

    embedding = extract_features(vectorizer, th_image[None, ...].to(device)).squeeze(0)
    output_batch = th.flatten(embedding, start_dim=1).T  # 49, 2048  
    
    response = beam_search(
        model=net, 
        source=output_batch[None, ...], 
        BOS=SPECIALS2IDX['<bos>'], 
        EOS=SPECIALS2IDX['<eos>'],
        max_len=64, 
        beam_width=beam_width,
        device=device, 
        alpha=0.7
    )
    
    logger.debug(f'nb generated : {len(response)}')
    sentences = []
    for sequence, _ in response:
        caption = vocab.lookup_tokens(sequence[1:-1])  # ignore <bos> and <eos>
        joined_caption = ' '.join(caption)
        sentences.append(joined_caption)
        
    logger.debug('ranking will begin...!')
    pil_image = cv2pil(cv_image)
    ranked_scores = rank_solutions(pil_image, sentences, ranker, processor, device)
    ranked_response = list(zip(sentences, ranked_scores))
    ranked_response = sorted(ranked_response, key=op.itemgetter(1), reverse=True)

    for caption, score in ranked_response:
        score = int(score * 100)
        logger.debug(f'caption : {caption} | score : {score:03d}')


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorizer', type=str, default='src/resnet152.th')
    parser.add_argument('--checkpoint_path', type=str, default='src/checkpoint_128.th')
    parser.add_argument('--image_path', type=str, default='images/bob.jpg')
    parser.add_argument('--vocabulary', type=str, default='src/vocabulary.pkl')
    parser.add_argument('--beam_width', type=int, default=17)
    parser.add_argument('--ranker', type=str, default='src/ranker.pkl')
    
    args = parser.parse_args()
    main(args)