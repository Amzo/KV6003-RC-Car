#!/usr/bin/env python3
import pygame

def initialize():
	pygame.init()
	pygame.camera.init()

	window = pygame.display.set_mode((640, 480))
	cam_list = pygame.camera.list_cameras()
	cam = pygame.camera.Camera(cam_list[0],(640,480))
	cam.start()

	return window, cam


def updateWindow(cam, window):
	frame = cam.get_image()
	frame = pygame.transform.scale(frame,(640,480))
	window.blit(frame,(0,0))
	pygame.display.update()


