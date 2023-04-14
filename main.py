import cv2
from aiogram import Bot, Dispatcher, executor

TOKEN = "6298126627:AAG5vCbi0viBECE7X6Pg7ixLND43z44wRdQ"

bot = Bot(token = TOKEN)
dp = Dispatcher(bot = bot)

def detect(photo):
	# считываем изображение
	img = cv2.imread(photo)

	# инициализация параметров детектирующей сети
	net = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg', 'yolov4-obj_best.weights')
	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

	# получаем найденные предметы и отрисовываем их
	classIds, scores, boxes = model.detect(img, confThreshold=0.2, nmsThreshold=0.01)  # 0.6 0.4 or 0.2 0.01
	for (classId, score, box) in zip(classIds, scores, boxes):
		print(classId)
		if classId == 0:
			color = (0, 255, 0)
		if classId == 1:
			color = (0, 0, 255)
		cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
					  color=color, thickness=2)

	# запись файла
	outfilename = 'out.jpg'
	cv2.imwrite(outfilename, img)


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
	await message.photo[-1].download('test.jpg')
	detect('test.jpg')
	photo = open('out.jpg', "rb")
	await bot.send_photo(message.from_user.id, photo)

if __name__ == '__main__':
	executor.start_polling(dp)


