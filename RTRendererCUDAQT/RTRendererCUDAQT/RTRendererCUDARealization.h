#pragma once
#include "RTRendererCUDAQT.h"
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include <QImage>
#include <QPainter>
#include <string>
#include <sstream>
#include <algorithm>

void LoopThread::run()
{
	kernel();
}


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



RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(10, 10, MAX_X + 400, MAX_Y + 400);

	looper = new LoopThread(this);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	logText = new QTextEdit(this);
	logText->setGeometry(MAX_X + 50, 20, 300, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(MAX_X + 220, MAX_Y + 40, 150, 20);

	ExitButton = new QPushButton("Exit", this);
	ExitButton->setGeometry(MAX_X + 220, MAX_Y + 80, 150, 20);

	PauseButton = new QPushButton("Pause", this);
	PauseButton->setGeometry(MAX_X + 220, MAX_Y + 120, 150, 20);

	ResetButton = new QPushButton("Default", this);
	ResetButton->setGeometry(MAX_X + 220, MAX_Y + 160, 150, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 220, MAX_Y + 200, 150, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 220, MAX_Y + 240, 150, 20);

	Parameter = new QComboBox(this);
	Parameter->setGeometry(MAX_X + 50, MAX_Y + 40, 150, 20);

	Parameter->addItem(QString::fromStdString("Parameter"));
	Parameter->addItem(QString::fromStdString("Camera"));
	Parameter->addItem(QString::fromStdString("World"));
	Parameter->addItem(QString::fromStdString("Render"));
	Parameter->addItem(QString::fromStdString("Object"));

	World = new QComboBox(this);
	World->setGeometry(MAX_X + 50, MAX_Y + 80, 150, 20);
	World->addItem(QString::fromStdString("ObjectList"));

	this->looper->object_names[0] = "deer";

	for(int i=0;i<OBJECT_NUM;i++)
		World->addItem(QString::fromStdString(this->looper->object_names[i]));

	lineObjectName = new QLineEdit(this);
	lineObjectName->setGeometry(20, MAX_Y + 20, 100, 20);

	labObjectName = new QLabel(this);
	labObjectName->setGeometry(20, MAX_Y + 40, 100, 20);
	labObjectName->setText(QString::fromStdString("ObjectName"));

	lineObjectStatus = new QLineEdit(this);
	lineObjectStatus->setGeometry(20, MAX_Y + 80, 100, 20);

	labObjectStatus = new QLabel(this);
	labObjectStatus->setGeometry(20, MAX_Y + 100, 100, 20);
	labObjectStatus->setText(QString::fromStdString("ObjectStatus"));

	AppearButton = new QPushButton(this);
	AppearButton->setGeometry(20, MAX_Y + 140, 100, 20);
	AppearButton->setText(QString::fromStdString("Appear"));

	DisappearButton = new QPushButton(this);
	DisappearButton->setGeometry(20, MAX_Y + 180, 100, 20);
	DisappearButton->setText(QString::fromStdString("Disappear"));




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

	

	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(ExitButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(PauseButton, SIGNAL(clicked()), this, SLOT(Pause()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(ResetButton, SIGNAL(clicked()), this, SLOT(Reset()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
	connect(Parameter, SIGNAL(activated(int)), this, SLOT(choosePara(int)));
	connect(World, SIGNAL(activated(int)), this, SLOT(chooseObject(int)));
	connect(DisappearButton, SIGNAL(clicked()), this, SLOT(disappear()));
	connect(AppearButton, SIGNAL(clicked()), this, SLOT(appear()));

	initialization();

}


void RTRendererCUDAQT::initialization()
{
	this->looper->break_flag = 0;
	this->looper->end_flag = 0;
	this->looper->pause_flag = 0;
	this->looper->targetClipUpperbound = 1.0;
	this->looper->targetSPP = INT_MAX;//默认是最大值，不依靠它提供break

	this->looper->Lookat = new Vec3(0, 0, 0);
	this->looper->Lookfrom = new Vec3(10, 5, 10);
	this->looper->Vup = new Vec3(0, 1, 0);

	this->looper->FocusDist = (*this->looper->Lookat - *this->looper->Lookfrom).length();
	this->looper->Aperture = 0.05;
	this->looper->Fov = 50.0;


	for (int i = 0; i < OBJECT_NUM; i++)
		this->looper->flag_show[i] = 1;

	discardParameter();
	hideAll();
}


void RTRendererCUDAQT::Pause()
{
	if (this->looper->pause_flag == 1)
		this->looper->pause_flag = 0;
	else
		this->looper->pause_flag = 1;
}

//根据下拉菜单返回的index决定显示的参数
void RTRendererCUDAQT::choosePara(int index)
{
	switch (index)
	{
	case 1:
		changeParaCamera();
		break;
	case 2:
		changeParaWorld();
		break;
	case 3:
		changeParaRender();
		break;
	case 4:
		changeObjectWorld();
		break;
	}
}

void RTRendererCUDAQT::disappear()
{
	this->looper->flag_show[this->selected] = 0;
	showObject();
	setParameter();
}

void RTRendererCUDAQT::appear()
{
	this->looper->flag_show[this->selected] = 1;
	showObject();
	setParameter();
}


void RTRendererCUDAQT::changeObjectWorld()
{
	hideAll();
	labObjectName->show();
	lineObjectName->show();
	labObjectStatus->show();
	lineObjectStatus->show();
	AppearButton->show();
	DisappearButton->show();
	World->show();

}


void RTRendererCUDAQT::showObject()
{
	lineObjectName->setText(this->World->currentText());
	if (this->looper->flag_show[this->selected])
		lineObjectStatus->setText(QString::fromStdString("Show"));
	else
		lineObjectStatus->setText(QString::fromStdString("Hidden"));
}


void RTRendererCUDAQT::chooseObject(int index)
{
	selected = index - 1;
	showObject();
	changeObjectWorld();
}





void RTRendererCUDAQT::setLabelCamera(int index, int x, int y, std::string name)
{
	labParaCamera[index] = new QLabel(this);
	lineParaCamera[index] = new  QLineEdit(this);
	labParaCamera[index]->setGeometry(x, y + 20, 100, 20);
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

	labObjectName->hide();
	lineObjectName->hide();
	labObjectStatus->hide();
	lineObjectStatus->hide();
	AppearButton->hide();
	DisappearButton->hide();
	World->hide();
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

void RTRendererCUDAQT::Reset()
{
	initialization();
	discardParameter();
	setParameter();
}

void RTRendererCUDAQT::checkParameterLegal()
{
	std::string str;

	if (this->lineParaRender[0]->text().toInt() <= 0)
	{
		this->lineParaRender[0]->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
		str = "targetSPP";
		str += " illegal !";
		this->logText->append(QString::fromStdString(str));
	}

	if (this->lineParaRender[1]->text().toFloat() <= 0)
	{
		this->lineParaRender[1]->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));
		str = "targetClipUpperbound";
		str += " illegal !";
		this->logText->append(QString::fromStdString(str));
	}

	//lookfrom, lookat 无法限制。只能希望用户别作死了

	
}

//设置参数。将窗口输入的数值对应赋值给各参数
void RTRendererCUDAQT::setParameter()
{
	checkParameterLegal();
	this->looper->targetSPP = this->lineParaRender[0]->text().toInt();
	this->looper->targetClipUpperbound = this->lineParaRender[1]->text().toFloat();

	this->looper->Lookat->e[0] = this->lineParaCamera[0]->text().toFloat();
	this->looper->Lookat->e[1] = this->lineParaCamera[1]->text().toFloat();
	this->looper->Lookat->e[2] = this->lineParaCamera[2]->text().toFloat();

	this->looper->Lookfrom->e[0] = this->lineParaCamera[3]->text().toFloat();
	this->looper->Lookfrom->e[1] = this->lineParaCamera[4]->text().toFloat();
	this->looper->Lookfrom->e[2] = this->lineParaCamera[5]->text().toFloat();

	this->looper->Vup->e[0] = this->lineParaCamera[6]->text().toFloat();
	this->looper->Vup->e[1] = this->lineParaCamera[7]->text().toFloat();
	this->looper->Vup->e[2] = this->lineParaCamera[8]->text().toFloat();

	//手动对焦  有输入参数
	if (this->lineParaCamera[9]->text().toFloat() != this->looper->FocusDist)
		this->looper->FocusDist = this->lineParaCamera[9]->text().toFloat();
	else
		this->looper->FocusDist = (*this->looper->Lookat - *this->looper->Lookfrom).length();


	this->looper->Aperture = this->lineParaCamera[10]->text().toFloat();
	this->looper->Fov = this->lineParaCamera[11]->text().toFloat();

	this->looper->placeHolder = this->lineParaWorld[0]->text().toFloat();

	this->ShowPara();
	this->looper->frameCount = 0;
	this->looper->break_flag = 1;

}

//在log显示参数
void RTRendererCUDAQT::ShowPara()
{
	std::string str = "SPP Threshold =" + std::to_string(this->looper->targetSPP);
	this->logText->append(QString::fromStdString(str));
	str = "Clip Upperbound =" + std::to_string(this->looper->targetClipUpperbound);
	this->logText->append(QString::fromStdString(str));

	str = "LookatX =" + std::to_string(this->looper->Lookat->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatY =" + std::to_string(this->looper->Lookat->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatZ =" + std::to_string(this->looper->Lookat->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromX =" + std::to_string(this->looper->Lookfrom->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromY =" + std::to_string(this->looper->Lookfrom->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromZ =" + std::to_string(this->looper->Lookfrom->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "VupX =" + std::to_string(this->looper->Vup->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "VupY =" + std::to_string(this->looper->Vup->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "VupZ =" + std::to_string(this->looper->Vup->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "FocusDist =" + std::to_string(this->looper->FocusDist);
	this->logText->append(QString::fromStdString(str));
	str = "Aperture =" + std::to_string(this->looper->Aperture);
	this->logText->append(QString::fromStdString(str));
	str = "Fov =" + std::to_string(this->looper->Fov);
	this->logText->append(QString::fromStdString(str));

	str = "Place Holder =" + std::to_string(this->looper->placeHolder);
	this->logText->append(QString::fromStdString(str));

	
	for (int i = 0; i < OBJECT_NUM; i++)
	{
		str = this->looper->object_names[i] + "'s status : ";
		if (this->looper->flag_show[i] == 1)
			str += "show";
		else
			str += "hidden";
	}
	
	this->logText->append(QString::fromStdString(str));

	this->discardParameter();

}

void RTRendererCUDAQT::discardParameter()
{
	this->lineParaRender[0]->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->lineParaRender[1]->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));

	this->lineParaCamera[0]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[0])));
	this->lineParaCamera[1]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[1])));
	this->lineParaCamera[2]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[2])));
	this->lineParaCamera[3]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[0])));
	this->lineParaCamera[4]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[1])));
	this->lineParaCamera[5]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[2])));
	this->lineParaCamera[6]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[0])));
	this->lineParaCamera[7]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[1])));
	this->lineParaCamera[8]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[2])));
	this->lineParaCamera[9]->setText(QString::fromStdString(std::to_string(this->looper->FocusDist)));
	this->lineParaCamera[10]->setText(QString::fromStdString(std::to_string(this->looper->Aperture)));
	this->lineParaCamera[11]->setText(QString::fromStdString(std::to_string(this->looper->Fov)));

	this->lineParaWorld[0]->setText(QString::fromStdString(std::to_string(this->looper->placeHolder)));
}

