from geoguessr import Game

game = Game()
game.play_rounds(1000, show=False, learn=True, save_after=50, show_after=-1)