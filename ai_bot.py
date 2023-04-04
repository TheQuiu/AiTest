import asyncio
import os

from dotenv import load_dotenv
import discord
from discord.ext import commands
from loguru import logger
from network.provider import Provider
from network.tokenizer.tokenizer import AiTokenizer
from network.trainer import Trainer
load_dotenv()
tokenizer = AiTokenizer()
trainer = Trainer(tokenizer)
provider = Provider(tokenizer)

bot = commands.Bot(command_prefix="%", intents=discord.Intents.all())
bot_dev = os.getenv("BOT_DEV")
bot_token = os.getenv("BOT_TOKEN")
@bot.event
async def on_ready():
    logger.info("Bot is ready!")


@bot.slash_command(name="predict", description="Predicts sentiment of the phrase")
async def predict(ctx: discord.ApplicationContext, phrase: str):
    logger.info(f"{ctx.author.name} {ctx.author.discriminator} asked for {phrase} prediction")
    await ctx.respond(provider.predict_sentiment(phrase))


@bot.slash_command(name="insert", description="Inserts phrase and value to dataset: 0 - negative, 1 - positive")
async def insert(ctx: discord.ApplicationContext, phrase: str, value: int):
    if ctx.author.id == bot_dev:
        if value == 1 or value == 0:
            with open('dataset/data.txt', 'a', encoding="UTF-8") as f:
                f.write(f"{phrase.lower()} % {value}\n")
            await ctx.respond(f"Inserted: {phrase} % {value}")
        else:
            await ctx.respond("You must enter 0 or 1 as value!")
    else:
        await ctx.respond("You are not allowed to do this!")


@bot.slash_command(name="start", description="Starts training model")
async def start(ctx: discord.ApplicationContext):
    if ctx.author.id == bot_dev:
        await ctx.respond("Training started!")
        task = asyncio.create_task(trainer.train())  # запускаем асинхронную функцию
        await task
        await callback_train(ctx)
        provider.reload_model()
    else:
        await ctx.respond("You are not allowed to do this!")


async def callback_train(ctx: discord.ApplicationContext):
    embed = discord.Embed(title='Model training result', color=0x9652f9)
    loss = trainer.get_history()["loss"]
    accuracy = trainer.get_history()["accuracy"]
    # Добавляем поля для каждой эпохи
    for i in range(len(loss)):
        epoch_num = i + 1
        embed.add_field(name=f'Epoch {epoch_num}', value=f'Loss: {loss[i]}\nAccuracy: {accuracy[i]}', inline=False)
    await ctx.respond(embed=embed)

bot.run(bot_token)
