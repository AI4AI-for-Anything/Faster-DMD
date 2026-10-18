"""SD1.5 DDP effective-batch updates; detached pre-update samples feed guidance."""
from contextlib import ExitStack, nullcontext
import torch


def _no_sync(module):
    return module.no_sync() if hasattr(module, 'no_sync') else nullcontext()


def _add_mean(totals, values, count):
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            totals[name] = totals.get(name, 0) + value / count


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite ' + label)


def advance_reference_scheduler(scheduler, reference_world_size):
    # Accelerate 0.23 advances the underlying scheduler world_size times for
    # each step when split_batches=False. Preserve the upstream 64-rank clock,
    # including generator-scheduler advancement on guidance-only iterations.
    raw = getattr(scheduler, 'scheduler', scheduler)
    for _ in range(reference_world_size):
        raw.step()


def accumulated_step(t):
    a = t.accelerator
    count = t.args.gradient_accumulation_steps
    update_g = t.step % t.dfake_gen_update_ratio == 0
    visual = t.step % t.wandb_iters == 0
    g_losses, d_losses, pending = {}, {}, []
    t.optimizer_generator.zero_grad(set_to_none=True)
    t.optimizer_guidance.zero_grad(set_to_none=True)
    generator_grad_norm = torch.zeros((), device=a.device)
    for micro in range(count):
        noise = torch.randn(t.batch_size, t.latent_channel, t.latent_resolution,
                            t.latent_resolution, device=a.device)
        tokens = next(t.dataloader if update_g else t.guidance_dataloader)
        tokens = tokens['text_input_ids_one'].squeeze(1)
        uncond = t.uncond_embedding.repeat(len(tokens), 1, 1)
        real = next(t.real_dataloader) if t.cls_on_clean_image else None
        with ExitStack() as stack:
            # G needs input gradients through guidance, never guidance parameter
            # gradients or a guidance reducer cycle in this phase.
            stack.enter_context(_no_sync(t.model.guidance_model))
            if micro < count - 1:
                stack.enter_context(_no_sync(t.model.feedforward_model))
            losses, g_log = t.model(
                noise, tokens, uncond, visual=visual and micro == count - 1,
                denoising_dict=None, real_train_dict=real,
                compute_generator_gradient=update_g,
                generator_turn=True, guidance_turn=False)
            if update_g:
                loss = 0
                if not t.args.gan_alone:
                    loss = loss + losses['loss_dm'] * t.args.dm_loss_weight
                if t.cls_on_clean_image and t.gen_cls_loss:
                    loss = loss + losses['gen_cls_loss'] * t.gen_cls_loss_weight
                _finite(loss.detach(), 'generator loss')
                a.backward(loss / count)
        _add_mean(g_losses, losses, count)
        # This dictionary contains detached generated latents, not saved graphs.
        # Guidance must use these PRE-generator-update images, as upstream does.
        pending.append((noise, tokens, uncond, real, g_log['guidance_data_dict']))
    if update_g:
        generator_grad_norm = a.clip_grad_norm_(t.model.feedforward_model.parameters(), t.max_grad_norm)
        _finite(generator_grad_norm, 'generator gradient norm')
        t.optimizer_generator.step()
        t.optimizer_generator.zero_grad(set_to_none=True)
    advance_reference_scheduler(t.scheduler_generator, t.args.schedule_world_size)
    for micro, (noise, tokens, uncond, real, generated) in enumerate(pending):
        context = _no_sync(t.model.guidance_model) if micro < count - 1 else nullcontext()
        with context:
            losses, d_log = t.model(
                noise, tokens, uncond, visual=visual and micro == count - 1,
                denoising_dict=None, real_train_dict=real,
                compute_generator_gradient=update_g,
                generator_turn=False, guidance_turn=True, guidance_data_dict=generated)
            loss = losses['loss_fake_mean']
            if t.cls_on_clean_image:
                loss = loss + losses['guidance_cls_loss'] * t.guidance_cls_loss_weight
            _finite(loss.detach(), 'guidance loss')
            a.backward(loss / count)
        _add_mean(d_losses, losses, count)
    guidance_grad_norm = a.clip_grad_norm_(t.model.guidance_model.parameters(), t.max_grad_norm)
    _finite(guidance_grad_norm, 'guidance gradient norm')
    t.optimizer_guidance.step()
    t.optimizer_guidance.zero_grad(set_to_none=True)
    advance_reference_scheduler(t.scheduler_guidance, t.args.schedule_world_size)
    return g_losses, d_losses, g_log, d_log, generator_grad_norm, guidance_grad_norm, noise, real
