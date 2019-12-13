#include "RTRendererCUDAQT.h"
#include <QImage>
#include <QPainter>
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include "consts.h"
#include <string>
#include <sstream>



//���沼����
//������ һһ��Ӧ Lab->Lab StartButton->StartButton StopButton ParameterText->ParameterText  Para_Stop->Para_Stop
//������

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
	StartButton->setGeometry(MAX_X + 50,MAX_Y + 40 ,100 ,20 );

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(MAX_X + 50, MAX_Y + 80 , 100, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 50, MAX_Y + 120, 100, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 50, MAX_Y + 160, 100, 20);


	
	



	paraTargetSPP = new QLineEdit(this);
	paraTargetSPP->setGeometry(20, MAX_Y + 20, 80, 20);

	paraClipUpperbound = new QLineEdit(this);
	paraClipUpperbound->setGeometry(120, MAX_Y + 20, 80, 20);

	paraLookatX = new QLineEdit(this);
	paraLookatX->setGeometry(20, MAX_Y + 60, 80, 20);

	paraLookatY = new QLineEdit(this);
	paraLookatY->setGeometry(120, MAX_Y + 60, 80, 20);

	paraLookatZ = new QLineEdit(this);
	paraLookatZ->setGeometry(220, MAX_Y + 60, 80, 20);
	


	labTargetSPP = new QLabel(this);
	labTargetSPP->setGeometry(20, MAX_Y + 40, 80, 20);
	labTargetSPP->setText("TargetSPP");


	labClipUpperbound = new QLabel(this);
	labClipUpperbound->setGeometry(120, MAX_Y + 40, 80, 20);
	labClipUpperbound->setText("ClipUpperbound");

	labLookatX = new QLabel(this);
	labLookatX->setGeometry(20, MAX_Y + 80, 80, 20);
	labLookatX->setText("LookatX");

	labLookatY = new QLabel(this);
	labLookatY->setGeometry(120, MAX_Y + 80, 80, 20);
	labLookatY->setText("LookatY");

	labLookatZ = new QLabel(this);
	labLookatZ->setGeometry(220, MAX_Y + 80, 80, 20);
	labLookatZ->setText("LookatZ");




	looper = new LoopThread(this);
	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
}



void LoopThread::run()
{
	kernel();
}


//���Լ��Ӹ�ģ�壬��������һЩ������
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


void RTRendererCUDAQT::initialization()
{
	this->looper->break_flag = 0;
	this->looper->end_flag = 0;
	this->looper->targetClipUpperbound = 1.0;
	this->looper->targetSPP = INT_MAX;//Ĭ�������ֵ�����������ṩbreak
	this->looper->lookatX = 0;
	this->looper->lookatY = 0;
	this->looper->lookatZ = 0;
	this->discardParameter();
}

void RTRendererCUDAQT::Startear()
{
	this->initialization();
	looper->start();
}

void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
	this->looper->end_flag = 1;
}

void RTRendererCUDAQT::setParameter()
{
	this->looper->targetSPP = this->paraTargetSPP->text().toInt();

	this->looper->targetClipUpperbound = this->paraClipUpperbound->text().toFloat();

	this->looper->lookatX = this->paraLookatX->text().toFloat();
	this->looper->lookatY = this->paraLookatY->text().toFloat();
	this->looper->lookatZ = this->paraLookatZ->text().toFloat();
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
	str = "LookatX =" + std::to_string(this->looper->lookatX);
	this->logText->append(QString::fromStdString(str));
	str = "LookatY =" + std::to_string(this->looper->lookatY);
	this->logText->append(QString::fromStdString(str));
	str = "LookatZ =" + std::to_string(this->looper->lookatZ);
	this->logText->append(QString::fromStdString(str));
}

void RTRendererCUDAQT::discardParameter()
{
	this->paraTargetSPP->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->paraClipUpperbound->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));
	this->paraLookatX->setText(QString::fromStdString(std::to_string(this->looper->lookatX)));
	this->paraLookatY->setText(QString::fromStdString(std::to_string(this->looper->lookatY)));
	this->paraLookatZ->setText(QString::fromStdString(std::to_string(this->looper->lookatZ)));
}
