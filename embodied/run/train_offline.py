"""Offline training loop for world model pretraining."""

import elements
import embodied
from tqdm import tqdm


def train_offline(make_agent, make_replay, make_stream, make_logger, args):
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    step = logger.step
    train_agg = elements.Agg()
    train_fps = elements.FPS()
    batch_steps = args.batch_size * args.batch_length

    should_log = embodied.LocalClock(args.log_every)
    should_save = embodied.LocalClock(args.save_every)

    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    carry = [agent.init_train(args.batch_size)]

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

    print(f'Offline training: {len(replay)} sequences, {args.steps} steps')

    min_seqs = args.batch_size
    if len(replay) < min_seqs:
        raise RuntimeError(
            f'Replay has {len(replay)} sequences but needs at least {min_seqs}. '
            f'Collect more demo data before pretraining.')

    pbar = tqdm(total=args.steps, initial=int(step), desc='Pretrain')
    while step < args.steps:
        batch = next(stream_train)
        carry[0], outs, mets = agent.train(carry[0], batch)
        train_fps.step(batch_steps)
        step.increment(batch_steps)
        train_agg.add(mets, prefix='train')

        if 'replay' in outs:
            replay.update(outs['replay'])

        if should_log(step):
            logger.add(train_agg.result())
            logger.add(replay.stats(), prefix='replay')
            logger.add({'fps/train': train_fps.result()})
            logger.write()

        if should_save(step):
            cp.save()

        pbar.update(batch_steps)

    cp.save()
    pbar.close()
    print('Pretraining complete')
