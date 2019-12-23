#include "RTRendererCUDAQT.h"
#include <QImage>
#include <QPainter>
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include "consts.h"
#include <string>
#include <sstream>
#include <algorithm>



//认真布局了
//改名字 一一对应 Lab->Lab StartButton->StartButton StopButton ParameterText->ParameterText  Para_Stop->Para_Stop
//函数名

RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(10, 10, MAX_X + 400, MAX_Y + 400);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	logText = new QTextEdit(this);
	logText->setGeometry(MAX_X + 50, 20, 300, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(MAX_X + 170,MAX_Y + 40 ,100 ,20 );

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(MAX_X + 170, MAX_Y + 80 , 100, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 170, MAX_Y + 120, 100, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 170, MAX_Y + 160, 100, 20);

	CameraButton = new QPushButton("Camera", this);
	CameraButton->setGeometry(MAX_X + 50, MAX_Y + 40, 100, 20);

	WorldButton = new QPushButton("World", this);
	WorldButton->setGeometry(MAX_X + 50, MAX_Y + 80, 100, 20);

	RenderButton = new QPushButton("Render", this);
	RenderButton->setGeometry(MAX_X + 50, MAX_Y + 120, 100, 20);



	setLabelRender(0, 20, MAX_Y + 40, "TargetSPP");
	setLabelRender(1, 140, MAX_Y + 40, "ClipUpperbound");

	setLabelCamera(0, 20, MAX_Y + 40, "LookatX");
	setLabelCamera(1, 140, MAX_Y + 40, "LookatY");
	setLabelCamera(2, 260, MAX_Y + 40, "LookatZ");
	setLabelCamera(3, 20, MAX_Y + 80, "LookfromX");
	setLabelCamera(4, 140, MAX_Y + 80, "LookfromY");
	setLabelCamera(5, 260, MAX_Y + 80, "LookfromZ");
	setLabelCamera(6, 20, MAX_Y + 120, "VupX");
	setLabelCamera(7, 140, MAX_Y + 120, "VupY");
	setLabelCamera(8, 260, MAX_Y + 120, "VupZ");
	setLabelCamera(9, 20, MAX_Y + 160, "FocusDist");
	setLabelCamera(10, 140, MAX_Y + 160, "Aperture");
	setLabelCamera(11, 260, MAX_Y + 160, "Fov");

	setLabelWorld(0, 20, MAX_Y + 40, "PlaceHolder");

	looper = new LoopThread(this);

	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
	connect(CameraButton, SIGNAL(clicked()), this, SLOT(changeParaCamera()));
	connect(WorldButton, SIGNAL(clicked()), this, SLOT(changeParaWorld()));
	connect(RenderButton, SIGNAL(clicked()), this, SLOT(changeParaRender()));


	initialization();

}

void RTRendererCUDAQT::initialization()
{
	this->looper->break_flag = 0;
	this->looper->end_flag = 0;
	this->looper->targetClipUpperbound = 1.0;
	this->looper->targetSPP = INT_MAX;//默认是最大值，不依靠它提供break

	this->looper->Lookat = para3(500, 500, 500);
	this->looper->Lookfrom = para3(5000, 2000, 4000);
	this->looper->Vup = para3(0,1,0);

	this->looper->FocusDist = 5900;
	this->looper->Aperture = 0.05;
	this->looper->Fov = 50.0;

	discardParameter();
	hideAll();
}

void LoopThread::run()
{
	kernel();
}


//给自己加个模板，还有其他一些东西。
void LoopThread::PrintMessege()
{
	emit info_flag();
}

void LoopThread::checkBreak()
{
	if (this->frameCount * SPP >= this->targetSPP)
	{
		this->break_flag = 1;
	}
}

void RTRendererCUDAQT::setLabelCamera(int index, int x, int y, std::string name)
{
	labParaCamera[index] = new QLabel(this);
	lineParaCamera[index] = new  QLineEdit(this);
	labParaCamera[index]->setGeometry(x, y+20, 100, 20);
	lineParaCamera[index]->setGeometry(x, y, 100, 20);
	labParaCamera[index]->setText(QString::fromStdString(name));
}

void RTRendererCUDAQT::setLabelRender(int index, int x, int y, std::string name)
{
	labParaRender[index] = new QLabel(this);
	lineParaRender[index] = new  QLineEdit(this);
	labParaRender[index]->setGeometry(x, y + 20, 100, 20);
	lineParaRender[index]->setGeometry(x, y, 100, 20);
	labParaRender[index]->setText(QString::fromStdString(name));
}

void RTRendererCUDAQT::setLabelWorld(int index, int x, int y, std::string name)
{
	labParaWorld[index] = new QLabel(this);
	lineParaWorld[index] = new  QLineEdit(this);
	labParaWorld[index]->setGeometry(x, y + 20, 100, 20);
	lineParaWorld[index]->setGeometry(x, y, 100, 20);
	labParaWorld[index]->setText(QString::fromStdString(name));
}




void RTRendererCUDAQT::hideAll()
{
	for (int i = 0; i < paraNumCamere; i++)
	{
		labParaCamera[i]->hide();
		lineParaCamera[i]->hide();
	}
	for (int i = 0; i < paraNumWorld; i++)
	{
		labParaWorld[i]->hide();
		lineParaWorld[i]->hide();
	}
	for (int i = 0; i < paraNumRender; i++)
	{
		labParaRender[i]->hide();
		lineParaRender[i]->hide();
	}
}

void RTRendererCUDAQT::changeParaCamera()
{
	changingNum = paraNumCamere;
	changingLab = labParaCamera;
	changingLine = lineParaCamera;
	changePara();
}

void RTRendererCUDAQT::changeParaWorld()
{
	changingNum = paraNumWorld;
	changingLab = labParaWorld;
	changingLine = lineParaWorld;
	changePara();
}

void RTRendererCUDAQT::changeParaRender()
{
	changingNum = paraNumRender;
	changingLab = labParaRender;
	changingLine = lineParaRender;
	changePara();
}

void RTRendererCUDAQT::changePara()
{
	hideAll();
	for (int i = 0; i < changingNum; i++)
	{
		changingLab[i]->show();
		changingLine[i]->show();
	}
}



void RTRendererCUDAQT::Startear()
{
	this->ShowPara();
	looper->start();
}

void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
	this->looper->end_flag = 1;
}

void RTRendererCUDAQT::setParameter()
{
	this->looper->targetSPP = this->lineParaRender[0]->text().toInt();
	this->looper->targetClipUpperbound = this->lineParaRender[1]->text().toFloat();

	this->looper->Lookat.e[0] = this->lineParaCamera[0]->text().toFloat();
	this->looper->Lookat.e[1] = this->lineParaCamera[1]->text().toFloat();
	this->looper->Lookat.e[2] = this->lineParaCamera[2]->text().toFloat();

	this->looper->Lookfrom.e[0] = this->lineParaCamera[3]->text().toFloat();
	this->looper->Lookfrom.e[1] = this->lineParaCamera[4]->text().toFloat();
	this->looper->Lookfrom.e[2] = this->lineParaCamera[5]->text().toFloat();

	this->looper->Vup.e[0] = this->lineParaCamera[6]->text().toFloat();
	this->looper->Vup.e[1] = this->lineParaCamera[7]->text().toFloat();
	this->looper->Vup.e[2] = this->lineParaCamera[8]->text().toFloat();

	this->looper->FocusDist = this->lineParaCamera[9]->text().toFloat();
	this->looper->Aperture = this->lineParaCamera[10]->text().toFloat();
	this->looper->Fov = this->lineParaCamera[11]->text().toFloat();

	this->looper->placeHolder = this->lineParaWorld[0]->text().toFloat();

	this->ShowPara();
	this->looper->frameCount = 0;
	this->looper->break_flag = 1;

}

void RTRendererCUDAQT::ShowPara()
{
	std::string str = "SPP Threshold =" + std::to_string(this->looper->targetSPP);
	this->logText->append(QString::fromStdString(str));
	str = "Clip Upperbound =" + std::to_string(this->looper->targetClipUpperbound);
	this->logText->append(QString::fromStdString(str));

	str = "LookatX =" + std::to_string(this->looper->Lookat.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatY =" + std::to_string(this->looper->Lookat.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatZ =" + std::to_string(this->looper->Lookat.e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromX =" + std::to_string(this->looper->Lookfrom.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromY =" + std::to_string(this->looper->Lookfrom.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromZ =" + std::to_string(this->looper->Lookfrom.e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "VupX =" + std::to_string(this->looper->Vup.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "VupY =" + std::to_string(this->looper->Vup.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "VupZ =" + std::to_string(this->looper->Vup.e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "FocusDist =" + std::to_string(this->looper->FocusDist);
	this->logText->append(QString::fromStdString(str));
	str = "Aperture =" + std::to_string(this->looper->Aperture);
	this->logText->append(QString::fromStdString(str));
	str = "Fov =" + std::to_string(this->looper->Fov);
	this->logText->append(QString::fromStdString(str));

	str = "Place Holder =" + std::to_string(this->looper->placeHolder);
	this->logText->append(QString::fromStdString(str));

}

void RTRendererCUDAQT::discardParameter()
{
	this->lineParaRender[0]->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->lineParaRender[1]->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));

	this->lineParaCamera[0]->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[0])));
	this->lineParaCamera[1]->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[1])));
	this->lineParaCamera[2]->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[2])));
	this->lineParaCamera[3]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[0])));
	this->lineParaCamera[4]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[1])));
	this->lineParaCamera[5]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[2])));
	this->lineParaCamera[6]->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[0])));
	this->lineParaCamera[7]->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[1])));
	this->lineParaCamera[8]->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[2])));
	this->lineParaCamera[9]->setText(QString::fromStdString(std::to_string(this->looper->FocusDist)));
	this->lineParaCamera[10]->setText(QString::fromStdString(std::to_string(this->looper->Aperture)));
	this->lineParaCamera[11]->setText(QString::fromStdString(std::to_string(this->looper->Fov)));

	this->lineParaWorld[0]->setText(QString::fromStdString(std::to_string(this->looper->placeHolder)));
}
